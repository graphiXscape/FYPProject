# Logo Similarity Thresholds Documentation

## Overview
This system uses two different similarity detection methods to ensure logo uniqueness during registration:

1. **DeepSVG Similarity** (Threshold: 0.98)
2. **Procrustes Similarity** (Threshold: 0.95)

## Similarity Methods

### 1. DeepSVG Similarity (Threshold: 0.98)
- **Method**: Uses a pre-trained deep learning model to extract 256-dimensional feature vectors from SVG logos
- **Threshold**: 0.98 (98% similarity)
- **Database**: Milvus vector database
- **Metric**: Cosine similarity
- **Behavior**: If similarity ≥ 0.98, registration is blocked

### 2. Procrustes Similarity (Threshold: 0.95)
- **Method**: Compares geometric shapes using Procrustes analysis
- **Threshold**: 0.95 (95% similarity)
- **Database**: MongoDB (parsed coordinates)
- **Metric**: Procrustes distance converted to similarity score
- **Behavior**: If similarity ≥ 0.95, registration is blocked

## Registration Flow

1. **File Upload**: User uploads SVG logo(s)
2. **DeepSVG Encoding**: Extract feature vector using DeepSVG model
3. **Algorithm Encoding**: Parse SVG paths and extract coordinates
4. **Similarity Check**:
   - Check DeepSVG similarity against existing logos in Milvus
   - Check Procrustes similarity against existing logos in MongoDB
   - If ANY method finds similarity above threshold, registration fails
5. **Database Insertion**: If all checks pass, insert into both databases

## API Endpoints

### 1. Register Logo
```
POST /api/register-logo
```
- Registers logo if it passes similarity checks
- Returns detailed information about failed registrations
- Includes information about similar logos found

### 2. Check Similarity (Single Logo)
```
POST /api/check-similarity
```
- Checks for similar logos without registering
- Useful for pre-registration validation of single logo
- Returns whether logo can be registered and details about similar logos

### 3. Check Bulk Similarity (Multiple Logos)
```
POST /api/check-bulk-similarity
```
- Checks multiple logos for similarity without registering
- Useful for pre-registration validation of bulk uploads
- Returns detailed results for each logo
- Supports bulk file uploads

## Response Format

### Success Response
```json
{
  "can_register": true,
  "message": "Logo can be registered. No similar logos found above threshold.",
  "thresholds": {
    "deepsvg": 0.98,
    "procrustes": 0.95
  },
  "highest_scores": {
    "deepsvg": 0.85,
    "procrustes": 0.72
  }
}
```

### Failure Response (Similarity Found)
```json
{
  "can_register": false,
  "message": "Logo cannot be registered. Found 2 similar logo(s) above threshold.",
  "similar_logos": [
    {
      "logo_id": "507f1f77bcf86cd799439011",
      "company_name": "Example Corp",
      "file_name": "logo.svg",
      "similarity_score": 0.99,
      "method": "DeepSVG"
    }
  ],
  "thresholds": {
    "deepsvg": 0.98,
    "procrustes": 0.95
  }
}
```

## Frontend Integration

The frontend now includes:
- **Single logo pre-check**: Check similarity for individual logos before registration
- **Bulk logo pre-check**: Check similarity for multiple logos at once
- **Enhanced bulk registration**: Better feedback for bulk uploads with partial success handling
- **Detailed notifications**: Shows which logos are similar and why
- **Visual feedback**: Color-coded notifications (success, warning, error)
- **Similarity details**: Displays company names, file names, and similarity scores
- **Progress indicators**: Loading states for all operations
- **Summary information**: Clear breakdown of successful vs failed registrations

## Threshold Configuration

To modify thresholds, update these values in `app_for_db_insertion.py`:

```python
# DeepSVG similarity threshold
if hit.distance >= 0.98:  # Change this value

# Procrustes similarity threshold  
if score >= 0.95:  # Change this value
```

## Best Practices

1. **Use Pre-check**: Always check similarity before attempting registration
2. **Review Similar Logos**: When similarity is found, review the similar logos to understand why
3. **Adjust Design**: If similarity is found, consider modifying the logo design
4. **Monitor Thresholds**: Periodically review threshold values based on business needs

## Troubleshooting

### Common Issues

1. **False Positives**: If legitimate logos are being blocked, consider lowering thresholds
2. **False Negatives**: If similar logos are being registered, consider raising thresholds
3. **Performance**: DeepSVG encoding may take time for complex SVGs
4. **Memory**: Large batches of logos may require more memory

### Debug Information

The system logs detailed information about:
- Similarity check results
- Failed registrations
- Database operations
- Error messages

Check the logs directory for detailed debugging information. 