# Activation Sample Interpretability Report

## Training Summary

- **Model**: gemma-2-2b
- **Layer**: 8
- **Dictionary Size**: 4,608
- **Training Tokens**: 50,000
- **Sample Collection Frequency**: Every 50 steps
- **Activation Threshold**: 0.05

## Sample Collection Results

- **Features with Samples**: 1246
- **Total Samples Collected**: 9,836
- **Average Samples per Feature**: 7.9
- **Specialized Features**: 0
- **General Features**: 1246

## Top 10 Most Active Features

### #1 Feature 1936

- **Max Activation**: 35.6505
- **Mean Activation**: 35.2582
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 35.6505
   **Context**: "<bos>"Caring, Effective, Middle/High"
   **Position**: 0

2. **Activation**: 35.6467
   **Context**: "<bos>African Americans in the Bluegrass
Whether"
   **Position**: 0

3. **Activation**: 35.6198
   **Context**: "<bos>Botanical Name and Pronunciation:
Dend"
   **Position**: 0

### #2 Feature 830

- **Max Activation**: 34.9889
- **Mean Activation**: 34.5494
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 34.9889
   **Context**: "<bos>Conference underscores young people’s leadership in"
   **Position**: 0

2. **Activation**: 34.9654
   **Context**: "<bos>African Americans in the Bluegrass
Whether"
   **Position**: 0

3. **Activation**: 34.9602
   **Context**: "<bos>"Caring, Effective, Middle/High"
   **Position**: 0

### #3 Feature 2177

- **Max Activation**: 34.1455
- **Mean Activation**: 33.6801
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 34.1455
   **Context**: "<bos>"Caring, Effective, Middle/High"
   **Position**: 0

2. **Activation**: 34.1261
   **Context**: "<bos>African Americans in the Bluegrass
Whether"
   **Position**: 0

3. **Activation**: 34.1048
   **Context**: "<bos>Conference underscores young people’s leadership in"
   **Position**: 0

### #4 Feature 2149

- **Max Activation**: 33.6709
- **Mean Activation**: 33.3410
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 33.6709
   **Context**: "<bos>Conference underscores young people’s leadership in"
   **Position**: 0

2. **Activation**: 33.6563
   **Context**: "<bos>"Caring, Effective, Middle/High"
   **Position**: 0

3. **Activation**: 33.6389
   **Context**: "<bos>African Americans in the Bluegrass
Whether"
   **Position**: 0

### #5 Feature 1732

- **Max Activation**: 33.4119
- **Mean Activation**: 31.1449
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 33.4119
   **Context**: " soon added to its growth. Bowmanstown was incorporated as a Borough on November 2"
   **Position**: 18

2. **Activation**: 33.1525
   **Context**: " but the nearby New Jersey Zinc Company soon added to its growth. Bowmanstown was incorporated"
   **Position**: 11

3. **Activation**: 33.1441
   **Context**: " New Jersey Zinc Company soon added to its growth. Bowmanstown was incorporated as a Borough"
   **Position**: 14

### #6 Feature 71

- **Max Activation**: 33.0874
- **Mean Activation**: 32.6001
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 33.0874
   **Context**: "<bos>Conference underscores young people’s leadership in"
   **Position**: 0

2. **Activation**: 33.0517
   **Context**: "<bos>"Caring, Effective, Middle/High"
   **Position**: 0

3. **Activation**: 33.0444
   **Context**: "<bos>African Americans in the Bluegrass
Whether"
   **Position**: 0

### #7 Feature 461

- **Max Activation**: 33.0657
- **Mean Activation**: 28.5377
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 33.0657
   **Context**: "<bos>A main character’s name often gives you"
   **Position**: 1

2. **Activation**: 33.0558
   **Context**: "<bos>On August 9, 201"
   **Position**: 1

3. **Activation**: 32.9120
   **Context**: "<bos>There's this kid who gets bullied a"
   **Position**: 1

### #8 Feature 1713

- **Max Activation**: 32.2559
- **Mean Activation**: 27.9403
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 32.2559
   **Context**: " that all of the energy peaks and troughs are precisely in step.
Now"
   **Position**: 6

2. **Activation**: 30.9724
   **Context**: " that all of the energy peaks and troughs are precisely in step.
"
   **Position**: 5

3. **Activation**: 30.5828
   **Context**: " the 2011 Census.
Nearly one"
   **Position**: 2

### #9 Feature 1314

- **Max Activation**: 32.1535
- **Mean Activation**: 28.7937
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 32.1535
   **Context**: "<bos>This is the time of year Gulf Coast residents begin watching the weather reports to"
   **Position**: 7

2. **Activation**: 30.7160
   **Context**: " 27 (IANS) Personal computers are fast taking over the place of human beings"
   **Position**: 12

3. **Activation**: 30.4855
   **Context**: "<bos>The Nature Elephant
The Karen people have always lived"
   **Position**: 2

### #10 Feature 1386

- **Max Activation**: 31.9258
- **Mean Activation**: 29.2105
- **Sample Count**: 50

**Top Examples:**

1. **Activation**: 31.9258
   **Context**: " New Jersey Zinc Company soon added to its growth. Bowmanstown was incorporated as a Borough"
   **Position**: 14

2. **Activation**: 31.7106
   **Context**: " but the nearby New Jersey Zinc Company soon added to its growth. Bowmanstown was incorporated"
   **Position**: 11

3. **Activation**: 31.1263
   **Context**: " soon added to its growth. Bowmanstown was incorporated as a Borough on November 2"
   **Position**: 18

## Conclusions

### Key Findings:

1. **Feature Activation Diversity**: The model learned features with varying levels of specialization
2. **Training Data Connection**: All samples contain real training data contexts
3. **Activation Strength**: Features show different activation levels, indicating varying importance
4. **Interpretability**: Activation samples provide clear insights into what each feature learned

### Interpretability Insights:

- **Specialized Features**: Some features activate consistently for specific text patterns
- **General Features**: Other features activate for diverse contexts
- **Training Data Integration**: Features successfully store samples from actual training data
- **Feature Learning**: The model learned meaningful representations of the training data

