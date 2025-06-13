
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from classify import RumourDetectClass
def test_with_official_data():
    detector = RumourDetectClass()
    test_df = pd.read_csv('./val.csv')  
    assert 'text' in test_df.columns and 'label' in test_df.columns

    predictions = []
    for text in test_df['text']:
        pred = detector.classify(text)
        predictions.append(pred)

    print("\n=== 官方测试集性能 ===")
    print(f"准确率: {accuracy_score(test_df['label'], predictions):.4f}")
    print(classification_report(test_df['label'], predictions, target_names=['非谣言', '谣言']))

    print("\n=== 样例对比 ===")
    sample_df = test_df.sample(8, random_state=42)
    for idx, row in sample_df.iterrows():
        print(f"\n文本: {row['text'][:80]}...")
        print(f"真实标签: {'谣言' if row['label'] == 1 else '非谣言'}")
        print(f"预测结果: {'谣言' if predictions[idx] == 1 else '非谣言'}")

if __name__ == '__main__':
    test_with_official_data()
