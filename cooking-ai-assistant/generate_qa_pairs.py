"""
Generate QA Pairs from Recipe Dataset for BERT Training
CS 6120 Cooking QA Project - Hour 2: Data Processing
"""

import os
import json
import pandas as pd
import re
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class QAPairGenerator:
    """Generate QA pairs from recipe data for training."""

    def __init__(self):
        self.qa_templates = {
            'ingredients': [
                "What ingredients are needed for {title}?",
                "What do I need to make {title}?",
                "What ingredients go into {title}?",
                "What's required for making {title}?",
                "List the ingredients for {title}.",
                "What do you need to cook {title}?"
            ],
            'directions': [
                "How do you make {title}?",
                "What are the steps for making {title}?",
                "How to prepare {title}?",
                "What's the recipe for {title}?",
                "How do I cook {title}?",
                "What are the cooking instructions for {title}?"
            ],
            'cooking_method': [
                "How is {title} prepared?",
                "What cooking method is used for {title}?",
                "How do you cook {title}?"
            ],
            'dish_type': [
                "What type of dish is {title}?",
                "What category does {title} belong to?",
                "What kind of food is {title}?"
            ]
        }

        # Patterns for extracting cooking information
        self.time_patterns = [
            r'(\d+)\s*(?:to\s*)?(\d+)?\s*(?:minutes?|mins?)',
            r'(\d+)\s*(?:to\s*)?(\d+)?\s*(?:hours?|hrs?)',
            r'(\d+)\s*(?:to\s*)?(\d+)?\s*(?:seconds?|secs?)'
        ]

        self.temp_patterns = [
            r'(\d+)(?:\s*to\s*(\d+))?\s*(?:degrees?\s*)?(?:F|Fahrenheit)',
            r'(\d+)(?:\s*to\s*(\d+))?\s*(?:degrees?\s*)?(?:C|Celsius)'
        ]

    def load_recipe_data(self, file_path: str) -> pd.DataFrame:
        """Load recipe data from file."""
        try:
            if file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)

            print(f"Loaded {len(df):,} recipes from {file_path}")
            return df

        except Exception as e:
            print(f" Error loading {file_path}: {e}")
            return pd.DataFrame()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?()-]', '', text)

        return text

    def extract_cooking_time(self, text: str) -> str:
        """Extract cooking time from recipe text."""
        text_lower = text.lower()

        for pattern in self.time_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        if match[1]:  # Range like "10 to 15 minutes"
                            return f"{match[0]} to {match[1]} minutes"
                        else:
                            return f"{match[0]} minutes"
                    else:
                        return f"{match} minutes"

        return "varies"

    def extract_temperature(self, text: str) -> str:
        """Extract cooking temperature from recipe text."""
        text_lower = text.lower()

        for pattern in self.temp_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        if match[1]:
                            return f"{match[0]} to {match[1]}°F"
                        else:
                            return f"{match[0]}°F"
                    else:
                        return f"{match}°F"

        return "varies"

    def identify_dish_type(self, title: str, ingredients: str) -> str:
        """Identify the type of dish based on title and ingredients."""
        title_lower = title.lower()
        ingredients_lower = ingredients.lower()

        # Simple classification based on keywords
        if any(word in title_lower for word in ['cookie', 'cake', 'pie', 'dessert', 'sweet']):
            return "dessert"
        elif any(word in title_lower for word in ['salad', 'green']):
            return "salad"
        elif any(word in title_lower for word in ['soup', 'stew', 'broth']):
            return "soup"
        elif any(word in title_lower for word in ['pasta', 'spaghetti', 'noodle']):
            return "pasta dish"
        elif any(word in ingredients_lower for word in ['chicken', 'beef', 'pork', 'fish']):
            return "main course"
        elif any(word in title_lower for word in ['breakfast', 'pancake', 'waffle']):
            return "breakfast"
        else:
            return "main dish"

    def generate_ingredient_qa(self, recipe: Dict) -> List[Dict]:
        """Generate QA pairs about ingredients."""
        qa_pairs = []
        title = recipe['title']
        ingredients = self.clean_text(str(recipe['ingredients']))

        if not ingredients or len(ingredients) < 10:
            return qa_pairs

        # Generate multiple question variations
        for template in self.qa_templates['ingredients']:
            question = template.format(title=title)

            qa_pairs.append({
                'question': question,
                'answer': ingredients,
                'context': f"Recipe: {title}. Ingredients: {ingredients}. Instructions: {self.clean_text(str(recipe.get('directions', '')))}",
                'type': 'ingredients',
                'title': title
            })

        return qa_pairs

    def generate_directions_qa(self, recipe: Dict) -> List[Dict]:
        """Generate QA pairs about cooking directions."""
        qa_pairs = []
        title = recipe['title']
        directions = self.clean_text(str(recipe.get('directions', '')))
        ingredients = self.clean_text(str(recipe['ingredients']))

        if not directions or len(directions) < 20:
            return qa_pairs

        # Generate multiple question variations
        for template in self.qa_templates['directions']:
            question = template.format(title=title)

            qa_pairs.append({
                'question': question,
                'answer': directions,
                'context': f"Recipe: {title}. Ingredients: {ingredients}. Instructions: {directions}",
                'type': 'directions',
                'title': title
            })

        return qa_pairs

    def generate_cooking_details_qa(self, recipe: Dict) -> List[Dict]:
        """Generate QA pairs about cooking details (time, temperature, etc.)."""
        qa_pairs = []
        title = recipe['title']
        directions = self.clean_text(str(recipe.get('directions', '')))
        ingredients = self.clean_text(str(recipe['ingredients']))

        # Extract cooking time
        cooking_time = self.extract_cooking_time(directions)
        if cooking_time != "varies":
            qa_pairs.append({
                'question': f"How long does it take to make {title}?",
                'answer': cooking_time,
                'context': f"Recipe: {title}. Instructions: {directions}",
                'type': 'cooking_time',
                'title': title
            })

            qa_pairs.append({
                'question': f"What's the cooking time for {title}?",
                'answer': cooking_time,
                'context': f"Recipe: {title}. Instructions: {directions}",
                'type': 'cooking_time',
                'title': title
            })

        # Extract temperature
        temperature = self.extract_temperature(directions)
        if temperature != "varies":
            qa_pairs.append({
                'question': f"What temperature should I cook {title} at?",
                'answer': temperature,
                'context': f"Recipe: {title}. Instructions: {directions}",
                'type': 'temperature',
                'title': title
            })

        # Dish type
        dish_type = self.identify_dish_type(title, ingredients)
        qa_pairs.append({
            'question': f"What type of dish is {title}?",
            'answer': dish_type,
            'context': f"Recipe: {title}. This is a {dish_type}.",
            'type': 'dish_type',
            'title': title
        })

        return qa_pairs

    def generate_step_by_step_qa(self, recipe: Dict) -> List[Dict]:
        """Generate QA pairs about specific steps."""
        qa_pairs = []
        title = recipe['title']
        directions = self.clean_text(str(recipe.get('directions', '')))

        if not directions or len(directions) < 50:
            return qa_pairs

        # Split directions into sentences/steps
        sentences = sent_tokenize(directions)

        if len(sentences) > 2:
            # Generate questions about specific steps
            for i, sentence in enumerate(sentences[:3]):  # First 3 steps
                if len(sentence.strip()) > 20:
                    qa_pairs.append({
                        'question': f"What is step {i + 1} for making {title}?",
                        'answer': sentence.strip(),
                        'context': f"Recipe: {title}. Full instructions: {directions}",
                        'type': 'step',
                        'title': title
                    })

        return qa_pairs

    def augment_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Augment QA pairs with variations and synonyms."""
        augmented_pairs = qa_pairs.copy()

        # Synonym replacements for common cooking terms
        synonyms = {
            'cook': ['prepare', 'make'],
            'ingredients': ['items', 'components'],
            'recipe': ['dish', 'meal'],
            'instructions': ['directions', 'steps', 'method'],
            'how to': ['how do I', 'what is the way to'],
            'what do I need': ['what ingredients are required', 'what do you need'],
            'how long': ['how much time', 'what is the cooking time']
        }

        # Create variations for a subset of QA pairs
        for qa in qa_pairs[:len(qa_pairs) // 3]:  # Augment 1/3 of pairs
            question = qa['question'].lower()

            for original, replacements in synonyms.items():
                if original in question:
                    for replacement in replacements:
                        new_question = question.replace(original, replacement)
                        # Capitalize first letter
                        new_question = new_question[0].upper() + new_question[1:]

                        augmented_pairs.append({
                            'question': new_question,
                            'answer': qa['answer'],
                            'context': qa['context'],
                            'type': qa['type'] + '_augmented',
                            'title': qa['title']
                        })
                    break  # Only apply first matching synonym

        return augmented_pairs

    def process_recipes(self, df: pd.DataFrame, max_recipes: int = 10000) -> List[Dict]:
        """Process recipes and generate QA pairs."""
        print(f" Generating QA pairs from {min(len(df), max_recipes):,} recipes...")

        all_qa_pairs = []

        # Process recipes with progress bar
        recipes_to_process = df.head(max_recipes)

        for idx, recipe in tqdm(recipes_to_process.iterrows(), total=len(recipes_to_process)):
            recipe_qa_pairs = []

            try:
                # Generate different types of QA pairs
                recipe_qa_pairs.extend(self.generate_ingredient_qa(recipe))
                recipe_qa_pairs.extend(self.generate_directions_qa(recipe))
                recipe_qa_pairs.extend(self.generate_cooking_details_qa(recipe))
                recipe_qa_pairs.extend(self.generate_step_by_step_qa(recipe))

                all_qa_pairs.extend(recipe_qa_pairs)

            except Exception as e:
                print(f"   Error processing recipe {idx}: {e}")
                continue

        print(f" Generated {len(all_qa_pairs):,} QA pairs from recipes")

        # Apply data augmentation
        print(" Applying data augmentation...")
        augmented_pairs = self.augment_qa_pairs(all_qa_pairs)
        print(f" Augmented to {len(augmented_pairs):,} QA pairs")

        return augmented_pairs

    def create_train_val_test_splits(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test splits."""
        print(" Creating train/validation/test splits...")

        # Shuffle the data
        random.shuffle(qa_pairs)

        # Split: 70% train, 15% val, 15% test
        total = len(qa_pairs)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)

        train_data = qa_pairs[:train_size]
        val_data = qa_pairs[train_size:train_size + val_size]
        test_data = qa_pairs[train_size + val_size:]

        print(f"   📊 Train: {len(train_data):,} pairs ({len(train_data) / total * 100:.1f}%)")
        print(f"   📊 Val: {len(val_data):,} pairs ({len(val_data) / total * 100:.1f}%)")
        print(f"   📊 Test: {len(test_data):,} pairs ({len(test_data) / total * 100:.1f}%)")

        return train_data, val_data, test_data

    def save_qa_pairs(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Save QA pairs to files."""
        print("💾 Saving QA pairs...")

        os.makedirs('data/processed', exist_ok=True)

        # Save train data
        with open('data/processed/train_qa_pairs.json', 'w') as f:
            json.dump(train_data, f, indent=2)

        # Save validation data
        with open('data/processed/val_qa_pairs.json', 'w') as f:
            json.dump(val_data, f, indent=2)

        # Save test data
        with open('data/processed/test_qa_pairs.json', 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"   ✅ Saved train_qa_pairs.json ({len(train_data):,} pairs)")
        print(f"   ✅ Saved val_qa_pairs.json ({len(val_data):,} pairs)")
        print(f"   ✅ Saved test_qa_pairs.json ({len(test_data):,} pairs)")

        # Create summary
        summary = {
            "total_qa_pairs": len(train_data) + len(val_data) + len(test_data),
            "train_pairs": len(train_data),
            "val_pairs": len(val_data),
            "test_pairs": len(test_data),
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_types": ["ingredients", "directions", "cooking_time", "temperature", "dish_type", "step"],
            "files": [
                "data/processed/train_qa_pairs.json",
                "data/processed/val_qa_pairs.json",
                "data/processed/test_qa_pairs.json"
            ]
        }

        with open('data/processed/qa_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"   📋 Summary saved: data/processed/qa_summary.json")

    def analyze_qa_pairs(self, qa_pairs: List[Dict]):
        """Analyze the generated QA pairs."""
        print(f"\n📊 QA Pairs Analysis:")

        # Count by type
        type_counts = {}
        for pair in qa_pairs:
            qa_type = pair['type']
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

        print(f"   QA pairs by type:")
        for qa_type, count in sorted(type_counts.items()):
            print(f"     {qa_type}: {count:,}")

        # Average lengths
        q_lengths = [len(pair['question']) for pair in qa_pairs]
        a_lengths = [len(pair['answer']) for pair in qa_pairs]
        c_lengths = [len(pair['context']) for pair in qa_pairs]

        print(f"   Average lengths:")
        print(f"     Questions: {sum(q_lengths) / len(q_lengths):.1f} characters")
        print(f"     Answers: {sum(a_lengths) / len(a_lengths):.1f} characters")
        print(f"     Context: {sum(c_lengths) / len(c_lengths):.1f} characters")

        # Show examples
        print(f"\n📝 Sample QA Pairs:")
        for i, pair in enumerate(qa_pairs[:3]):
            print(f"\n   Example {i + 1} ({pair['type']}):")
            print(f"   Q: {pair['question']}")
            print(f"   A: {pair['answer'][:100]}...")


def main():
    """Main function to generate QA pairs."""
    print("🍳 Generating QA Pairs for CS 6120 BERT Training")
    print("=" * 60)

    # Initialize generator
    generator = QAPairGenerator()

    # Load recipe data
    possible_files = [
        'data/raw/recipes_sample_10k.json',
        'data/raw/recipes_full.json',
        'data/raw/recipenlg_sample_50k.json'
    ]

    df = pd.DataFrame()
    for file_path in possible_files:
        if os.path.exists(file_path):
            df = generator.load_recipe_data(file_path)
            break

    if df.empty:
        print("❌ No recipe data found. Please run download script first.")
        return False

    # Generate QA pairs (limit to 10k recipes for reasonable processing time)
    qa_pairs = generator.process_recipes(df, max_recipes=10000)

    if not qa_pairs:
        print("❌ No QA pairs generated")
        return False

    # Analyze generated pairs
    generator.analyze_qa_pairs(qa_pairs)

    # Create train/val/test splits
    train_data, val_data, test_data = generator.create_train_val_test_splits(qa_pairs)

    # Save data
    generator.save_qa_pairs(train_data, val_data, test_data)

    print(f"\n🎉 SUCCESS! QA pairs generated and ready for BERT training!")
    print(f"Total QA pairs: {len(qa_pairs):,}")
    print(f"Files saved in data/processed/")
    print(f"Next: Set up BERT fine-tuning pipeline")

    return True


if __name__ == "__main__":
    main()