"""
Enhanced LLM Service with Extraction, Generation, and Validation Capabilities
"""

import logging
from typing import List, Dict, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import Config

logger = logging.getLogger(__name__)


class LLMService:
    """Enhanced LLM service for extraction, generation, and Q&A"""
    
    def __init__(self, vector_store):
        """
        Initialize LLM service
        
        Args:
            vector_store: VectorStore instance for retrieval
        """
        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=4096
        )
        
        # For structured extraction (lower temperature)
        self.extraction_llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=4096
        )
        
        # For generation (higher temperature for creativity)
        self.generation_llm = ChatOpenAI(
            temperature=0.8,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=4096
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.vector_store = vector_store
        
        try:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vector_store.vector_store.as_retriever(),
                memory=self.memory
            )
        except Exception as e:
            logger.warning(f"Failed to initialize retrieval chain: {e}")
            self.chain = None

    def _invoke_model(self, model, prompt: str) -> str:
        """Invoke a chat model across LangChain API variants and normalize the text response."""
        try:
            return model.predict(prompt)
        except (AttributeError, TypeError):
            response = model.invoke(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)

    def get_response(self, query: str) -> str:
        """
        Get response to a query (backward compatible)
        
        Args:
            query: User query
            
        Returns:
            LLM response
        """
        try:
            if self.chain:
                response = self.chain({"question": query})
                return response['answer']
            else:
                print("Retrieval chain not initialized, using direct LLM response")
                print(f"Query: {query}")
                return self._invoke_model(self.llm, query)
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return "I encountered an error processing your request."

    def invoke_extraction(self, prompt: str) -> str:
        """
        Invoke LLM for extraction with structured output
        
        Args:
            prompt: Extraction prompt
            
        Returns:
            LLM response with extracted entities
        """
        try:
            print(f"Invoking extraction with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.extraction_llm, prompt)
            print(f"Extraction response:\n{response}\n{'='*50}")
            logger.debug(f"Extraction completed. Response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

    def generate_requirements_document(self, entities: List[Dict]) -> str:
        """
        Generate a requirements document from extracted entities
        
        Args:
            entities: List of requirement entities
            
        Returns:
            Generated requirements document
        """
        prompt = f"""Based on the following extracted requirements, generate a comprehensive requirements document.
        
Requirements:
{self._format_entities(entities)}

Generate a professional requirements document with:
1. Executive Summary
2. Functional Requirements (organized by category)
3. Non-Functional Requirements
4. Constraints and Assumptions
5. Success Criteria

Format as markdown."""

        try:
            print(f"Generating requirements document with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.generation_llm, prompt)
            print(f"Generated requirements document:\n{response}\n{'='*50}")
            logger.info("Generated requirements document")
            return response
        except Exception as e:
            logger.error(f"Error generating requirements: {e}")
            raise

    def generate_design_document(self, entities: List[Dict]) -> str:
        """
        Generate a design document from extracted entities
        
        Args:
            entities: List of design and entity specification
            
        Returns:
            Generated design document
        """
        prompt = f"""Based on the following extracted entities and design patterns, generate a comprehensive design document.

Entities and Patterns:
{self._format_entities(entities)}

Generate a professional design document with:
1. Architecture Overview
2. Component Design
3. Data Models
4. API Design (if applicable)
5. Integration Points
6. Deployment Considerations

Format as markdown."""

        try:
            print(f"Generating design document with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.generation_llm, prompt)
            print(f"Generated design document:\n{response}\n{'='*50}")
            logger.info("Generated design document")
            return response
        except Exception as e:
            logger.error(f"Error generating design document: {e}")
            raise

    def generate_test_cases(self, requirement_entities: List[Dict]) -> str:
        """
        Generate test cases from requirements
        
        Args:
            requirement_entities: List of requirement entities
            
        Returns:
            Generated test cases
        """
        prompt = f"""Based on the following requirements, generate comprehensive test cases.

Requirements:
{self._format_entities(requirement_entities)}

Generate test cases organized by:
1. Functional Tests (covering each requirement)
2. Non-Functional Tests (performance, security, etc.)
3. Edge Cases and Error Scenarios
4. Integration Tests

For each test case include: ID, Name, Description, Preconditions, Steps, Expected Result."""

        try:
            print(f"Generating test cases with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.generation_llm, prompt)
            print(f"Generated test cases:\n{response}\n{'='*50}")
            logger.info("Generated test cases")
            return response
        except Exception as e:
            logger.error(f"Error generating test cases: {e}")
            raise

    def generate_rules_document(self, rule_entities: List[Dict]) -> str:
        """
        Generate a business rules document
        
        Args:
            rule_entities: List of rule entities
            
        Returns:
            Generated rules document
        """
        prompt = f"""Based on the following extracted business rules, generate a comprehensive business rules document.

Rules:
{self._format_entities(rule_entities)}

Generate a professional business rules document with:
1. Overview and Purpose
2. Business Rules organized by domain/scope
3. Constraints and Limitations
4. Rule Dependencies and Interactions
5. Enforcement Mechanisms

Format as markdown."""

        try:
            print(f"Generating business rules document with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.generation_llm, prompt)
            print(f"Generated business rules document:\n{response}\n{'='*50}")
            logger.info("Generated business rules document")
            return response
        except Exception as e:
            logger.error(f"Error generating rules document: {e}")
            raise

    def validate_consistency(self, text1: str, text2: str) -> bool:
        """
        Validate if two texts are consistent
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if consistent, False if contradictory
        """
        prompt = f"""Are these two statements logically consistent with each other?

Statement 1: {text1}
Statement 2: {text2}

Answer with only: CONSISTENT or CONTRADICTORY"""

        try:
            print(f"Validating consistency with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.extraction_llm, prompt)
            print(f"Consistency validation response:\n{response}\n{'='*50}")
            return "CONSISTENT" in response.upper()
        except Exception as e:
            logger.warning(f"Consistency validation error: {e}")
            return True  # Default to consistent if validation fails

    def invoke_for_validation(self, prompt: str) -> str:
        """
        Invoke LLM for validation checks
        
        Args:
            prompt: Validation prompt
            
        Returns:
            Validation response
        """
        try:
            print(f"Invoking validation with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.extraction_llm, prompt)
            print(f"Validation response:\n{response}\n{'='*50}")
            return response
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise

    def suggest_improvements(self, content: str) -> List[str]:
        """
        Suggest improvements for extracted content
        
        Args:
            content: Content to analyze
            
        Returns:
            List of improvement suggestions
        """
        prompt = f"""Analyze the following knowledge/requirements content and suggest 3-5 improvements:

{content}

Provide suggestions for:
1. Clarity and completeness
2. Consistency with standards
3. Missing critical information
4. Format and organization

List each suggestion as a numbered item."""

        try:
            print(f"Generating improvement suggestions with prompt:\n{prompt}\n{'='*50}")
            response = self._invoke_model(self.generation_llm, prompt)
            print(f"Improvement suggestions response:\n{response}\n{'='*50}")
            # Parse response into list
            suggestions = [line.strip() for line in response.split('\n') if line.strip() and any(char.isdigit() for char in line[:3])]
            logger.info(f"Generated {len(suggestions)} improvement suggestions")
            return suggestions
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

    def _format_entities(self, entities: List[Dict]) -> str:
        """Format entities for prompt input"""
        formatted = []
        for entity in entities:
            text = entity.get('text', '') or entity.get('canonical_text', '')
            entity_type = entity.get('type', '') or entity.get('entity_type', '')
            attrs = entity.get('attributes', {}) or entity.get('unified_attributes', {})
            
            entry = f"- [{entity_type}] {text}"
            if attrs:
                entry += f"\n  Properties: {attrs}"
            formatted.append(entry)
        
        return "\n".join(formatted) if formatted else "No entities provided"

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.debug("Conversation memory cleared")