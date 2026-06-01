from src.training_evaluation import ComprehensiveEvaluator
from src.cooking_qa_system import CookingQASystem

qa_system = CookingQASystem()
qa_system.initialize_system()

evaluator = ComprehensiveEvaluator(qa_system)
results = evaluator.run_comprehensive_evaluation()

report = evaluator.generate_evaluation_report(results)
with open("results/evaluation_report.md", "w") as f:
    f.write(report)

print("✅ Evaluation complete! Check results/evaluation_report.md")

