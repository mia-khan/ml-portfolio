"""
Cooking QA System
A domain-specific Question Answering system for cooking and meal preparation.
Includes BERT-QA, TF-IDF retrieval, meal planning, and grocery list generation.
"""

import json
import pickle
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer
)
import torch.nn.functional as F

# NLP libraries
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataPreprocessor:
    """Handles data loading and preprocessing for cooking datasets."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def load_recipe_data(self, file_path: str) -> pd.DataFrame:
        """Load recipe data from JSON or CSV format."""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return sample data for testing
            return self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample cooking data for testing."""
        sample_recipes = [
            {
                "title": "Classic Chocolate Chip Cookies",
                "ingredients": ["2 cups flour", "1 cup butter", "1 cup sugar", "2 eggs", "1 cup chocolate chips"],
                "instructions": "Preheat oven to 375°F. Mix butter and sugar. Add eggs and flour. Fold in chocolate chips. Bake for 10-12 minutes.",
                "cooking_time": "25 minutes",
                "difficulty": "easy"
            },
            {
                "title": "Spaghetti Carbonara",
                "ingredients": ["1 lb spaghetti", "6 eggs", "1 cup parmesan", "8 oz pancetta", "black pepper"],
                "instructions": "Cook spaghetti. Fry pancetta. Mix eggs and cheese. Combine hot pasta with egg mixture. Add pancetta and pepper.",
                "cooking_time": "20 minutes",
                "difficulty": "medium"
            },
            {
                "title": "Garden Salad",
                "ingredients": ["mixed greens", "tomatoes", "cucumber", "carrots", "olive oil", "vinegar"],
                "instructions": "Wash and chop vegetables. Mix greens and vegetables. Dress with oil and vinegar.",
                "cooking_time": "10 minutes",
                "difficulty": "easy"
            }
        ]
        return pd.DataFrame(sample_recipes)

    def create_qa_pairs(self, recipes_df: pd.DataFrame) -> List[Dict]:
        """Create SQuAD-format QA pairs from recipe data."""
        qa_pairs = []

        for _, recipe in recipes_df.iterrows():
            context = f"Recipe: {recipe['title']}. Ingredients: {', '.join(recipe['ingredients']) if isinstance(recipe['ingredients'], list) else recipe['ingredients']}. Instructions: {recipe['instructions']}. Cooking time: {recipe['cooking_time']}."

            # Generate various question types
            questions = [
                {
                    "question": f"What ingredients are needed for {recipe['title']}?",
                    "answer": ", ".join(recipe['ingredients']) if isinstance(recipe['ingredients'], list) else recipe[
                        'ingredients'],
                    "context": context
                },
                {
                    "question": f"How long does it take to make {recipe['title']}?",
                    "answer": recipe['cooking_time'],
                    "context": context
                },
                {
                    "question": f"How do you make {recipe['title']}?",
                    "answer": recipe['instructions'],
                    "context": context
                }
            ]
            qa_pairs.extend(questions)

        return qa_pairs

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words]

        return ' '.join(tokens)


class CookingBERTQA:
    """BERT-based Question Answering model fine-tuned for cooking domain."""

    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_training_data(self, qa_pairs: List[Dict]) -> Dict:
        """Prepare data for training/fine-tuning."""
        encodings = []
        for qa in qa_pairs:
            encoded = self.tokenizer(
                qa['question'],
                qa['context'],
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            # Find answer positions (simplified for demo)
            answer_start = qa['context'].lower().find(qa['answer'].lower())
            answer_end = answer_start + len(qa['answer']) if answer_start != -1 else 0

            encoded['start_positions'] = torch.tensor([answer_start])
            encoded['end_positions'] = torch.tensor([answer_end])
            encodings.append(encoded)

        return encodings

    def answer_question(self, question: str, context: str) -> Dict:
        """Generate answer for a given question and context."""
        inputs = self.tokenizer(
            question, context,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely answer
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        if end_idx < start_idx:
            end_idx = start_idx

        answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        confidence = float((F.softmax(start_scores, dim=1)[0][start_idx] *
                            F.softmax(end_scores, dim=1)[0][end_idx]))

        return {
            "answer": answer,
            "confidence": confidence,
            "start_idx": start_idx.item(),
            "end_idx": end_idx.item()
        }


class TFIDFRetrieval:
    """TF-IDF based retrieval system for cooking recipes."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.recipe_vectors = None
        self.recipes_df = None
        self.preprocessor = DataPreprocessor()

    def build_index(self, recipes_df: pd.DataFrame):
        """Build TF-IDF index from recipe corpus."""
        self.recipes_df = recipes_df

        # Combine recipe information for indexing
        corpus = []
        for _, recipe in recipes_df.iterrows():
            text = f"{recipe['title']} {recipe['ingredients']} {recipe['instructions']}"
            processed_text = self.preprocessor.preprocess_text(text)
            corpus.append(processed_text)

        self.recipe_vectors = self.vectorizer.fit_transform(corpus)
        print(f"Built TF-IDF index with {len(corpus)} recipes")

    def retrieve_relevant_recipes(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant recipes for a query."""
        if self.recipe_vectors is None:
            raise ValueError("Index not built. Call build_index first.")

        # Process query
        processed_query = self.preprocessor.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.recipe_vectors).flatten()

        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            recipe = self.recipes_df.iloc[idx]
            results.append({
                "recipe": recipe.to_dict(),
                "similarity_score": similarities[idx],
                "context": f"Recipe: {recipe['title']}. Ingredients: {recipe['ingredients']}. Instructions: {recipe['instructions']}."
            })

        return results


class IngredientExtractor:
    """Extract and process ingredients from text."""

    def __init__(self):
        # Load spaCy model (use smaller model if large one not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: en_core_web_sm not found. Using basic NLP processing.")
            self.nlp = None

        # Common cooking units and measurements
        self.units = {
            'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon',
            'teaspoons', 'tsp', 'pound', 'pounds', 'lb', 'lbs', 'ounce',
            'ounces', 'oz', 'gram', 'grams', 'g', 'kilogram', 'kg', 'liter', 'l'
        }

        # Common ingredients
        self.common_ingredients = {
            'flour', 'sugar', 'butter', 'eggs', 'milk', 'salt', 'pepper',
            'chicken', 'beef', 'pork', 'fish', 'tomato', 'onion', 'garlic',
            'oil', 'rice', 'pasta', 'cheese', 'bread', 'potato', 'carrot'
        }

    def extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients from text."""
        ingredients = []

        if self.nlp:
            doc = self.nlp(text.lower())
            for token in doc:
                if (token.text in self.common_ingredients or
                        token.pos_ == "NOUN" and len(token.text) > 2):
                    ingredients.append(token.text)
        else:
            # Fallback: simple keyword matching
            words = text.lower().split()
            for word in words:
                word_clean = re.sub(r'[^a-zA-Z]', '', word)
                if word_clean in self.common_ingredients:
                    ingredients.append(word_clean)

        return list(set(ingredients))  # Remove duplicates

    def normalize_ingredient(self, ingredient: str) -> str:
        """Normalize ingredient names."""
        # Remove quantities and units
        words = ingredient.lower().split()
        normalized_words = []

        for word in words:
            if (not word.isdigit() and
                    word not in self.units and
                    len(word) > 2):
                normalized_words.append(word)

        return ' '.join(normalized_words)


class MealPlanner:
    """Meal planning and recipe recommendation system."""

    def __init__(self):
        self.ingredient_extractor = IngredientExtractor()
        self.intent_keywords = {
            'recipe_question': ['recipe', 'how to make', 'cook', 'bake', 'prepare'],
            'ingredient_query': ['ingredient', 'need', 'require', 'use'],
            'meal_planning': ['meal', 'plan', 'suggest', 'recommend', 'what to eat'],
            'grocery_list': ['grocery', 'shopping', 'buy', 'store', 'list']
        }

    def classify_intent(self, query: str) -> str:
        """Classify user intent based on query."""
        query_lower = query.lower()

        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score

        # Return intent with highest score, default to recipe_question
        return max(intent_scores, key=intent_scores.get) if max(intent_scores.values()) > 0 else 'recipe_question'

    def recommend_recipes(self, available_ingredients: List[str],
                          recipes_df: pd.DataFrame,
                          dietary_restrictions: Optional[List[str]] = None) -> List[Dict]:
        """Recommend recipes based on available ingredients."""
        recommendations = []

        # Normalize available ingredients
        normalized_available = [
            self.ingredient_extractor.normalize_ingredient(ing)
            for ing in available_ingredients
        ]

        for _, recipe in recipes_df.iterrows():
            # Extract ingredients from recipe
            recipe_ingredients = self.ingredient_extractor.extract_ingredients(
                str(recipe['ingredients'])
            )

            # Calculate ingredient match score
            normalized_recipe_ingredients = [
                self.ingredient_extractor.normalize_ingredient(ing)
                for ing in recipe_ingredients
            ]

            matches = sum(1 for ing in normalized_recipe_ingredients
                          if ing in normalized_available)
            total_ingredients = len(normalized_recipe_ingredients)

            if total_ingredients > 0:
                match_score = matches / total_ingredients

                recommendations.append({
                    'recipe': recipe.to_dict(),
                    'match_score': match_score,
                    'matched_ingredients': matches,
                    'total_ingredients': total_ingredients
                })

        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:5]  # Return top 5


class GroceryListGenerator:
    """Generate consolidated grocery shopping lists."""

    def __init__(self):
        self.ingredient_extractor = IngredientExtractor()

        # Grocery store categories
        self.categories = {
            'produce': ['tomato', 'onion', 'garlic', 'lettuce', 'carrot', 'potato', 'apple', 'banana'],
            'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream', 'eggs'],
            'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'lamb'],
            'pantry': ['flour', 'sugar', 'salt', 'pepper', 'oil', 'vinegar', 'rice', 'pasta'],
            'bakery': ['bread', 'rolls', 'bagels']
        }

    def extract_recipe_ingredients(self, recipes: List[Dict]) -> List[str]:
        """Extract all ingredients from selected recipes."""
        all_ingredients = []

        for recipe in recipes:
            ingredients_text = recipe.get('ingredients', '')
            if isinstance(ingredients_text, list):
                ingredients_text = ' '.join(ingredients_text)

            recipe_ingredients = self.ingredient_extractor.extract_ingredients(ingredients_text)
            all_ingredients.extend(recipe_ingredients)

        return all_ingredients

    def consolidate_shopping_list(self, ingredient_lists: List[List[str]]) -> Dict:
        """Merge and consolidate ingredients into organized shopping list."""
        # Flatten all ingredients
        all_ingredients = []
        for ingredient_list in ingredient_lists:
            all_ingredients.extend(ingredient_list)

        # Normalize and count
        ingredient_counts = {}
        for ingredient in all_ingredients:
            normalized = self.ingredient_extractor.normalize_ingredient(ingredient)
            ingredient_counts[normalized] = ingredient_counts.get(normalized, 0) + 1

        # Categorize ingredients
        categorized_list = {category: [] for category in self.categories.keys()}
        categorized_list['other'] = []

        for ingredient, count in ingredient_counts.items():
            categorized = False
            for category, items in self.categories.items():
                if any(item in ingredient for item in items):
                    categorized_list[category].append({
                        'ingredient': ingredient,
                        'count': count
                    })
                    categorized = True
                    break

            if not categorized:
                categorized_list['other'].append({
                    'ingredient': ingredient,
                    'count': count
                })

        # Remove empty categories
        return {k: v for k, v in categorized_list.items() if v}


class CookingQASystem:
    """Main cooking QA system integrating all components."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.bert_qa = CookingBERTQA()
        self.tfidf_retrieval = TFIDFRetrieval()
        self.meal_planner = MealPlanner()
        self.grocery_generator = GroceryListGenerator()
        self.recipes_df = None

    def initialize_system(self, recipe_data_path: Optional[str] = None):
        """Initialize the system with recipe data."""
        # Load recipe data
        if recipe_data_path:
            self.recipes_df = self.preprocessor.load_recipe_data(recipe_data_path)
        else:
            self.recipes_df = self.preprocessor._create_sample_data()

        # Build TF-IDF index
        self.tfidf_retrieval.build_index(self.recipes_df)

        print(f"System initialized with {len(self.recipes_df)} recipes")

    def process_query(self, user_query: str) -> Dict:
        """Main processing pipeline for user queries."""
        if self.recipes_df is None:
            self.initialize_system()

        # Classify intent
        intent = self.meal_planner.classify_intent(user_query)

        response = {
            "query": user_query,
            "intent": intent,
            "response": None,
            "additional_info": {}
        }

        if intent == "recipe_question":
            response.update(self._handle_recipe_qa(user_query))
        elif intent == "meal_planning":
            response.update(self._handle_meal_planning(user_query))
        elif intent == "grocery_list":
            response.update(self._handle_grocery_generation(user_query))
        else:
            response.update(self._handle_recipe_qa(user_query))  # Default fallback

        return response

    def _handle_recipe_qa(self, question: str) -> Dict:
        """Handle recipe-related questions."""
        # Get relevant recipes using TF-IDF
        relevant_recipes = self.tfidf_retrieval.retrieve_relevant_recipes(question, top_k=3)

        if not relevant_recipes:
            return {
                "response": "I couldn't find relevant recipes for your question.",
                "method": "tfidf_retrieval"
            }

        # Use BERT QA on most relevant recipe
        best_recipe = relevant_recipes[0]
        bert_answer = self.bert_qa.answer_question(question, best_recipe['context'])

        return {
            "response": bert_answer['answer'] if bert_answer['confidence'] > 0.1 else best_recipe['context'],
            "confidence": bert_answer['confidence'],
            "relevant_recipes": [r['recipe']['title'] for r in relevant_recipes],
            "method": "bert_qa"
        }

    def _handle_meal_planning(self, query: str) -> Dict:
        """Handle meal planning requests."""
        # Extract ingredients from query
        available_ingredients = self.meal_planner.ingredient_extractor.extract_ingredients(query)

        if not available_ingredients:
            return {
                "response": "Please specify what ingredients you have available for meal planning.",
                "method": "meal_planning"
            }

        # Get recommendations
        recommendations = self.meal_planner.recommend_recipes(
            available_ingredients, self.recipes_df
        )

        if not recommendations:
            return {
                "response": "I couldn't find recipes matching your available ingredients.",
                "method": "meal_planning"
            }

        # Format response
        response_text = "Based on your available ingredients, I recommend:\n"
        for i, rec in enumerate(recommendations[:3], 1):
            recipe = rec['recipe']
            match_score = rec['match_score']
            response_text += f"{i}. {recipe['title']} (Match: {match_score:.0%})\n"

        return {
            "response": response_text,
            "recommendations": recommendations[:3],
            "available_ingredients": available_ingredients,
            "method": "meal_planning"
        }

    def _handle_grocery_generation(self, query: str) -> Dict:
        """Handle grocery list generation."""
        # For demo, extract recipe names from query or use top recipes
        recipe_names = []
        for _, recipe in self.recipes_df.iterrows():
            if recipe['title'].lower() in query.lower():
                recipe_names.append(recipe.to_dict())

        if not recipe_names:
            # Use first 2 recipes as example
            recipe_names = self.recipes_df.head(2).to_dict('records')

        # Generate grocery list
        all_ingredients = self.grocery_generator.extract_recipe_ingredients(recipe_names)
        shopping_list = self.grocery_generator.consolidate_shopping_list([all_ingredients])

        # Format response
        response_text = "Here's your grocery shopping list:\n\n"
        for category, items in shopping_list.items():
            if items:
                response_text += f"{category.title()}:\n"
                for item in items:
                    response_text += f"  - {item['ingredient']}\n"
                response_text += "\n"

        return {
            "response": response_text,
            "shopping_list": shopping_list,
            "recipes_used": [r['title'] for r in recipe_names],
            "method": "grocery_generation"
        }


# Evaluation functions
class SystemEvaluator:
    """Evaluate system performance."""

    def __init__(self, qa_system: CookingQASystem):
        self.qa_system = qa_system

    def calculate_em_f1(self, predictions: List[str], targets: List[str]) -> Dict:
        """Calculate Exact Match and F1 scores."""
        em_score = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip()) / len(predictions)

        # Simple F1 calculation based on word overlap
        f1_scores = []
        for pred, target in zip(predictions, targets):
            pred_words = set(pred.lower().split())
            target_words = set(target.lower().split())

            if len(pred_words) == 0 or len(target_words) == 0:
                f1_scores.append(0)
                continue

            precision = len(pred_words & target_words) / len(pred_words)
            recall = len(pred_words & target_words) / len(target_words)

            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)

        return {
            "exact_match": em_score,
            "f1_score": np.mean(f1_scores)
        }

    def evaluate_system(self, test_queries: List[Dict]) -> Dict:
        """Comprehensive system evaluation."""
        predictions = []
        targets = []

        for query_data in test_queries:
            query = query_data['question']
            expected = query_data['expected_answer']

            result = self.qa_system.process_query(query)
            predictions.append(result['response'])
            targets.append(expected)

        metrics = self.calculate_em_f1(predictions, targets)
        return metrics


# Demo and testing function
def demo_cooking_qa_system():
    """Demonstrate the cooking QA system functionality."""
    print("=== Cooking QA System Demo ===\n")

    # Initialize system
    qa_system = CookingQASystem()
    qa_system.initialize_system()

    # Test queries
    test_queries = [
        "What ingredients do I need for chocolate chip cookies?",
        "How long does it take to make spaghetti carbonara?",
        "I have eggs, cheese, and pasta. What can I make?",
        "Generate a grocery list for making cookies and salad",
        "How do you make carbonara?"
    ]

    print("Testing various queries:\n")
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        result = qa_system.process_query(query)
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response']}")
        print(f"Method: {result.get('method', 'unknown')}")
        print("-" * 50)

    return qa_system


if __name__ == "__main__":
    # Run demo
    system = demo_cooking_qa_system()

    print("\nSystem is ready for interactive use!")
    print("You can now use system.process_query('your question') to ask cooking questions.")