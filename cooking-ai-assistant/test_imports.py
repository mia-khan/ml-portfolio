try:
    import torch
    import transformers
    import sklearn
    import spacy
    import nltk
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
