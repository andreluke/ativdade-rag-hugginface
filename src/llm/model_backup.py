import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Optional

class HuggingFaceLLM:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Carregando modelo {model_name} em {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Configurar pad_token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Modelo carregado com sucesso
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, prompt, max_length=200):
        """
        Geração focada: primeiro tenta DialoGPT, se falhar usa RAG puro
        """
        # Extrair pergunta do usuário primeiro
        user_question = self._extract_user_question(prompt)
        
        # Verificar se é sobre DSM ANTES de processar
        if not self._is_dsm_question(user_question):
            return self._get_scope_warning()
        
        # Tentar DialoGPT apenas se modelo está funcionando
        if self.model and self.tokenizer:
            try:
                response = self._try_dialogpt_generation(prompt, max_length)
                if response and self._is_valid_response(response):
                    return self._polish_response(response)
            except Exception as e:
                print(f"DialoGPT falhou: {e}")
        
        # Fallback: RAG puro
        return self._get_rag_pure_response(prompt)
    
    def _try_dialogpt_generation(self, prompt: str, max_length: int) -> Optional[str]:
        """Tentativa limpa de gerar com DialoGPT"""
        # Verificar se modelo está disponível
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            # Preparar prompt otimizado para DialoGPT
            context_info = self._extract_clean_context(prompt)
            user_question = self._extract_user_question(prompt)
            
            # Prompt conversacional simples
            if context_info:
                conversation = f"Sobre mobile: {context_info[:200]}\nUsuário: {user_question}\nBot:"
            else:
                conversation = f"Usuário: {user_question}\nBot:"
            
            # Tokenizar
            inputs = self.tokenizer(
                conversation,
                return_tensors='pt',
                truncation=True,
                max_length=400,
                padding=True
            )
        
            input_ids = inputs['input_ids'].to(torch.device(self.device))
            attention_mask = inputs['attention_mask'].to(torch.device(self.device))
            
            # Gerar
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_length, 100),
                    num_beams=2,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
            # Extrair apenas resposta nova
            generated_tokens = outputs[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip() if response else None
            
        except Exception as e:
            print(f"Erro no DialoGPT: {e}")
            return None
    
    def _is_valid_response(self, response: str) -> bool:
        """Validação rigorosa de qualidade"""
        if not response or len(response.strip()) < 10:
            return False
        
        response_clean = response.strip().lower()
        
        # Padrões de nonsense comuns
        nonsense_patterns = [
            'pupupu', 'lalala', 'hahaha', 'jejeje', 'xoxoxo',
            'meu o que', 'estava a ou', 'según', 'híbrido pwa',
            '!!!!!!', '??????', '.......',
            'íticas:', 'púpúpú'
        ]
        
        for pattern in nonsense_patterns:
            if pattern in response_clean:
                return False
        
        # Verificar repetição excessiva
        words = response_clean.split()
        if len(words) > 5:
            # Se mais de 40% das palavras são repetidas, é problemático
            unique_words = set(words)
            if len(unique_words) < len(words) * 0.6:
                return False
        
        # Deve ter pelo menos algumas palavras relacionadas a tech/mobile
        tech_words = [
            'react', 'native', 'flutter', 'dart', 'javascript', 'mobile',
            'app', 'desenvolvimento', 'framework', 'componente', 'teste',
            'performance', 'android', 'ios', 'arquitetura'
        ]
        
        has_tech_content = any(word in response_clean for word in tech_words)
        
        # Aceitar se tem conteúdo tech OU é substancial (>50 chars)
        return has_tech_content or len(response.strip()) > 50
    
    def _polish_response(self, response: str) -> str:
        """Polimento final da resposta"""
        # Limpar quebras de linha excessivas
        polished = re.sub(r'\n\s*\n', '\n\n', response.strip())
        
        # Remover espaços extras
        polished = re.sub(r'\s+', ' ', polished)
        
        # Capitalizar primeira letra
        if polished and not polished[0].isupper():
            polished = polished[0].upper() + polished[1:]
        
        # Garantir pontuação adequada
        if polished and not polished.endswith(('.', '!', '?', ':')):
            polished += '.'
        
        return polished
    
    def _extract_user_question(self, prompt: str) -> str:
        """Extrai pergunta do usuário do prompt"""
        lines = prompt.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('Usuário:'):
                return line.replace('Usuário:', '').strip()
        return ""
    
    def _extract_clean_context(self, prompt: str) -> str:
        """Extrai contexto RAG limpo"""
        contexts = []
        
        if "Informações relevantes:" in prompt:
            lines = prompt.split('\n')
            in_context = False
            
            for line in lines:
                line = line.strip()
                if line == "Informações relevantes:":
                    in_context = True
                    continue
                elif in_context and line.startswith('•'):
                    raw_context = line[1:].strip()
                    clean_context = self._deep_clean_context(raw_context)
                    if len(clean_context) > 25:
                        contexts.append(clean_context)
                elif in_context and line and not line.startswith('Usuário:'):
                    break
        
        return ' '.join(contexts[:2]) if contexts else ""
    
    def _is_dsm_question(self, question: str) -> bool:
        """Verifica se a pergunta é sobre Desenvolvimento de Software Mobile"""
        question_lower = question.lower()
        
        dsm_keywords = [
            # Frameworks mobile
            'react native', 'flutter', 'ionic', 'xamarin', 'cordova', 'phonegap',
            # Plataformas
            'android', 'ios', 'mobile', 'app', 'aplicativo', 'multiplataforma',
            # Tecnologias mobile
            'expo', 'metro', 'gradle', 'xcode', 'fastlane', 'apk', 'ipa',
            # Conceitos mobile
            'responsivo', 'push notification', 'deep link', 'offline', 'sqlite',
            # Testes mobile
            'detox', 'appium', 'maestro', 'e2e mobile', 'device testing',
            # Performance mobile
            'fps', 'battery', 'memory mobile', 'startup time', 'bundle size',
            # Deploy mobile
            'play store', 'app store', 'testflight', 'play console', 'code push',
            # Arquiteturas mobile
            'mvvm mobile', 'clean mobile', 'repository pattern mobile'
        ]
        
        return any(keyword in question_lower for keyword in dsm_keywords)
    
    def _get_scope_warning(self) -> str:
        """Retorna mensagem de aviso sobre escopo DSM"""
        return ("🚫 Desculpe, sou especializado apenas em **Desenvolvimento de Software Mobile (DSM)**.\n\n"
               "Posso ajudar com:\n"
               "• **Frameworks:** React Native, Flutter, Ionic\n"
               "• **Arquiteturas móveis:** MVVM, Clean Architecture\n"
               "• **Testes:** Jest, Detox, Appium, E2E\n"
               "• **CI/CD:** GitHub Actions, Fastlane, CodePush\n"
               "• **Performance:** Otimizações iOS/Android\n"
               "• **Deploy:** App Store, Google Play\n\n"
               "Faça uma pergunta sobre desenvolvimento mobile! 📱")
    
    def _get_rag_pure_response(self, prompt: str) -> str:
        """Processa resposta usando apenas RAG puro"""
        # Extrair contextos do prompt
        contexts = []
        if "Informações relevantes:" in prompt:
            lines = prompt.split('\n')
            in_context_section = False
            for line in lines:
                if line.strip() == "Informações relevantes:":
                    in_context_section = True
                    continue
                elif in_context_section and line.startswith('•'):
                    contexts.append(line[1:].strip())
                elif in_context_section and line.strip() and not line.startswith('•'):
                    break
        
        if contexts:
            # Limpar contextos 
            clean_contexts = []
            for context in contexts:
                clean_context = self._deep_clean_context(context)
                if len(clean_context) > 30:
                    clean_contexts.append(clean_context)
            
            if clean_contexts:
                # Retornar resposta RAG pura
                combined_context = " ".join(clean_contexts[:2])
                return combined_context
        
        return ("Não encontrei informações específicas sobre isso na minha base de conhecimento DSM.\n\n"
               "Posso ajudar com React Native, Flutter, Ionic, arquiteturas móveis, testes, CI/CD e performance mobile.\n\n"
               "Você pode reformular a pergunta ou ser mais específico sobre qual framework ou aspecto mobile te interessa?")
    
    def _deep_clean_context(self, context: str) -> str:
        """Limpeza profunda de contexto RAG"""
        if not context or len(context.strip()) < 10:
            return ""
        
        clean_text = context.strip()
        
        # Remover padrões específicos de títulos e formatação
        patterns_to_remove = [
            r'^[A-Z\s]+$',  # Linhas só em maiúscula
            r'^={3,}.*={3,}$',  # Linhas com ===
            r'^\*+\s*.*\s*\*+$',  # Linhas com asteriscos
            r'^#+\s*',  # Headers markdown
            r'^\s*[-•]\s*$',  # Bullets vazios
        ]
        
        lines = clean_text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Verificar se linha deve ser removida
            should_remove = False
            for pattern in patterns_to_remove:
                if re.match(pattern, line):
                    should_remove = True
                    break
            
            if not should_remove and len(line) > 15:
                # Limpar prefixos de lista
                if line.startswith('- '):
                    line = line[2:].strip()
                elif line.startswith('• '):
                    line = line[2:].strip()
                
                clean_lines.append(line)
        
        result = ' '.join(clean_lines).strip()
        
        # Remover duplicações e espaços extras
        result = re.sub(r'\s+', ' ', result)
        
        return result
    
    def _clean_rag_context(self, context: str) -> str:
        """Limpa contexto RAG removendo títulos e formatação desnecessária"""
        # Remove títulos em maiúscula comuns (mais abrangente)
        titles_to_remove = [
            'COMPARISON:', 'REACT NATIVE VS FLUTTER ARQUITETURA:',
            'OTIMIZAÇÃO DE PERFORMANCE EM FLUTTER:', 'OTIMIZAÇÃO DE PERFORMANCE EM REACT NATIVE:',
            'CI/CD PARA REACT NATIVE - GUIA COMPLETO:', 'CI/CD E DEPLOYMENT PARA DESENVOLVIMENTO MOBILE:',
            'TESTING EM REACT NATIVE:', 'TESTES PARA APLICAÇÕES MOBILE:', 'ESTRATÉGIAS DE TESTE:',
            'ARQUITETURAS MOBILE:', 'CLEAN ARCHITECTURE:', 'MVVM PATTERN:',
            'CI/CD MULTIPLATAFORMA - FERRAMENTAS GERAIS:', 'CURVA DE APRENDIZADO:', 'Curva de Aprendizado:',
            'FERRAMENTAS DE BUILD:', 'PERFORMANCE:', 'DEPLOYMENT:', 'TESTING:'
        ]
        
        clean_context = context
        
        # Remove títulos específicos
        for title in titles_to_remove:
            clean_context = clean_context.replace(title, '')
        
        # Remove padrões de títulos em maiúscula seguidos de dois pontos
        clean_context = re.sub(r'[A-Z\s]{10,}:', '', clean_context)
        
        # Remove padrões menores que podem ser títulos também
        clean_context = re.sub(r'^[A-Z][a-zA-Z\s]{5,}:', '', clean_context, flags=re.MULTILINE)
        
        # Remove linhas com apenas símbolos ou muito curtas
        lines = clean_context.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Filtros mais rigorosos
            if (line and 
                not line.startswith('===') and 
                not line.startswith('---') and
                len(line) > 20 and  # Mínimo 20 caracteres
                not re.match(r'^[A-Z\s]{5,}:?$', line) and  # Não é título em maiúscula
                ':' not in line[:15]):  # Não tem dois pontos no início (título)
                
                # Remove bullets
                if line.startswith('- '):
                    line = line[2:]
                
                # Remove números de lista
                line = re.sub(r'^\d+\.\s*', '', line)
                
                clean_lines.append(line)
        
        # Junta as linhas limpas
        result = ' '.join(clean_lines).strip()
        
        # Remove espaços duplos
        result = re.sub(r'\s+', ' ', result)
        
        return result
    
    def _get_simple_fallback(self, prompt: str) -> str:
        """Fallback focado exclusivamente em DSM usando RAG com processamento natural"""
        # Extrair contextos do prompt se existirem
        contexts = []
        user_question = ""
        
        if "Informações relevantes:" in prompt:
            lines = prompt.split('\n')
            in_context_section = False
            for line in lines:
                if line.strip() == "Informações relevantes:":
                    in_context_section = True
                    continue
                elif in_context_section and line.startswith('•'):
                    contexts.append(line[1:].strip())
                elif in_context_section and line.strip() and not line.startswith('•'):
                    break
                    
        # Extrair pergunta do usuário para melhor contexto
        lines = prompt.split('\n')
        for line in reversed(lines):
            if line.startswith('Usuário:'):
                user_question = line.replace('Usuário:', '').strip()
                break
        
        # Verificar se a pergunta é sobre DSM (Desenvolvimento de Software Mobile)
        question_lower = user_question.lower()
        
        # Palavras-chave relacionadas ao DSM
        dsm_keywords = [
            # Frameworks mobile
            'react native', 'flutter', 'ionic', 'xamarin', 'cordova', 'phonegap',
            # Plataformas
            'android', 'ios', 'mobile', 'app', 'aplicativo', 'multiplataforma',
            # Tecnologias mobile
            'expo', 'metro', 'gradle', 'xcode', 'fastlane', 'apk', 'ipa',
            # Conceitos mobile
            'responsivo', 'push notification', 'deep link', 'offline', 'sqlite',
            # Testes mobile
            'detox', 'appium', 'maestro', 'e2e mobile', 'device testing',
            # Performance mobile
            'fps', 'battery', 'memory mobile', 'startup time', 'bundle size',
            # Deploy mobile
            'play store', 'app store', 'testflight', 'play console', 'code push',
            # Arquiteturas mobile
            'mvvm mobile', 'clean mobile', 'repository pattern mobile',
            # Geral mobile
            'cicd', 'ci/cd', 'deploy', 'build', 'teste', 'testing', 'performance', 'otimização'
        ]
        
        # Verificar se a pergunta contém palavras-chave de DSM
        is_dsm_related = any(keyword in question_lower for keyword in dsm_keywords)
        
        # Se não for sobre DSM, alertar o usuário
        if not is_dsm_related and not contexts:
            return ("🚫 Desculpe, sou especializado apenas em **Desenvolvimento de Software Mobile (DSM)**.\n\n"
                   "Posso ajudar com:\n"
                   "• **Frameworks:** React Native, Flutter, Ionic\n"
                   "• **Arquiteturas móveis:** MVVM, Clean Architecture\n"
                   "• **Testes:** Jest, Detox, Appium, E2E\n"
                   "• **CI/CD:** GitHub Actions, Fastlane, CodePush\n"
                   "• **Performance:** Otimizações iOS/Android\n"
                   "• **Deploy:** App Store, Google Play\n\n"
                   "Faça uma pergunta sobre desenvolvimento mobile! 📱")
        
        # Se há contexto do RAG, processar de forma natural
        if contexts:
            # Limpar contextos de títulos e formatação
            clean_contexts = []
            for context in contexts:
                clean_context = self._clean_rag_context(context)
                if len(clean_context) > 30:  # Contexto substancial
                    clean_contexts.append(clean_context)
            
            if clean_contexts:
                # Retornar contexto limpo diretamente (mais natural)
                context_info = " ".join(clean_contexts[:2])  # Máximo 2 contextos
                return context_info
        
        # Se não há contexto mas é sobre DSM, indicar que pode ajudar
        if is_dsm_related:
            return ("Não encontrei informações específicas sobre isso na minha base de conhecimento DSM.\n\n"
                   "Posso ajudar com React Native, Flutter, Ionic, arquiteturas móveis, testes, CI/CD e performance mobile.\n\n"
                   "Você pode reformular a pergunta ou ser mais específico sobre qual framework ou aspecto mobile te interessa?")
        
        # Fallback final
        return ("🚫 Por favor, faça perguntas sobre **Desenvolvimento de Software Mobile**.\n\n"
               "Especialidades: React Native, Flutter, Ionic, arquiteturas, testes e CI/CD mobile. 📱")