def calculate_sequence_accuracy(predictions, targets):
    if not targets:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets) 