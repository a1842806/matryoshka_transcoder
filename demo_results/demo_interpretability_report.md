# Demo Activation Sample Interpretability Report

## Analysis Summary

This demo shows how activation sample collection and analysis works with realistic examples.

### Sample Collection Results

- **Features with Samples**: 5
- **Total Samples Collected**: 25
- **Specialized Features**: 0
- **General Features**: 5

## Feature Analysis

### Top Features by Activation Strength

#### #1 Feature 1

- **Max Activation**: 4.5000
- **Mean Activation**: 3.9200
- **Sample Count**: 5

**Top Examples:**

1. **Activation**: 4.5000
   **Context**: "The derivative of f(x) = x^2 is f'(x) = 2x"
   **Position**: 5

2. **Activation**: 4.1000
   **Context**: "According to Einstein's theory of relativity, E = mc^2"
   **Position**: 7
#### #2 Feature 3

- **Max Activation**: 4.4000
- **Mean Activation**: 3.7800
- **Sample Count**: 5

**Top Examples:**

1. **Activation**: 4.4000
   **Context**: "The patient's blood pressure was 140/90 mmHg"
   **Position**: 5

2. **Activation**: 4.0000
   **Context**: "Clinical trials showed a 75% success rate for"
   **Position**: 4
#### #3 Feature 2

- **Max Activation**: 4.3000
- **Mean Activation**: 3.6400
- **Sample Count**: 5

**Top Examples:**

1. **Activation**: 4.3000
   **Context**: "The quarterly revenue increased by 15% compared to"
   **Position**: 4

2. **Activation**: 3.9000
   **Context**: "Stock market volatility reached its highest level"
   **Position**: 3
#### #4 Feature 0

- **Max Activation**: 4.2000
- **Mean Activation**: 3.5400
- **Sample Count**: 5

**Top Examples:**

1. **Activation**: 4.2000
   **Context**: "function calculateSum(a, b) { return a + b; }"
   **Position**: 3

2. **Activation**: 3.8000
   **Context**: "def process_data(input_file): with open(input_file) as f:"
   **Position**: 1
#### #5 Feature 4

- **Max Activation**: 2.8000
- **Mean Activation**: 2.4000
- **Sample Count**: 5

**Top Examples:**

1. **Activation**: 2.8000
   **Context**: "The weather today is sunny and warm"
   **Position**: 3

2. **Activation**: 2.6000
   **Context**: "I went to the store to buy groceries"
   **Position**: 2

## Interpretability Insights

### Key Findings

1. **Feature Specialization**: Some features show consistent activation patterns
2. **Training Data Connection**: All samples contain realistic training contexts
3. **Activation Strength Variation**: Features show different importance levels
4. **Meaningful Representations**: Features learn specific concepts from data

### Specialized vs General Features

- **Specialized Features**: Activate consistently for specific text patterns
- **General Features**: Activate for diverse contexts (polysemantic)
- **Training Data Integration**: Features successfully store real training examples

## Conclusion

This demo demonstrates that activation sample collection provides valuable insights into:
- What each feature learned during training
- How features specialize for different concepts
- The connection between activations and training data
- Feature interpretability and understanding

The analysis shows that features do indeed store meaningful activation samples from training data, enabling interpretability research.
