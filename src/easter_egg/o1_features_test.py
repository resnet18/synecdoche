def extract_o1_features(text):
    features = {
        "char_count": len(text),
        "punct_count": sum(1 for c in text if c in "，。！？；："),
        "has_parentheses": 1 if "（" in text or "(" in text else 0,
        "has_ellipsis": 1 if "..." in text or "…" in text else 0,
    }
    return features

test_sentence = "数据量500句，单卡16G，还想发ACL？（做梦吧）"
result = extract_o1_features(test_sentence)

print("特征工程测试：")
for key, value in result.items():
    print(f"  {key}: {value}")