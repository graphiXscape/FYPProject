import React, { useState } from 'react';
import './RegisterLogo.css';

const RegisterLogo = () => {
  const [formData, setFormData] = useState({
    companyName: '',
    websiteURL: '',
    metadata: '',
    logoFiles: [],
  });

  const [previewURLs, setPreviewURLs] = useState([]);
  const [allowMultiple, setAllowMultiple] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState(null);
  const [isCheckingBulkSimilarity, setIsCheckingBulkSimilarity] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setNotification(null);

    if (!formData.logoFiles.length) {
      setNotification({ type: 'error', message: 'Please upload at least one SVG logo.' });
      setIsLoading(false);
      return;
    }

    const invalidFiles = formData.logoFiles.filter(file => file.type !== 'image/svg+xml');
    if (invalidFiles.length > 0) {
      setNotification({ type: 'error', message: 'Please make sure all uploaded files are SVG images.' });
      setIsLoading(false);
      return;
    }

    const data = new FormData();
    data.append('companyName', formData.companyName);
    data.append('websiteURL', formData.websiteURL);
    data.append('metadata', formData.metadata);

    formData.logoFiles.forEach((file) => {
      data.append('logos', file);
    });

    try {
      const response = await fetch('http://localhost:5000/api/register-logo', {
        method: 'POST',
        body: data,
      });

      const result = await response.json();
      console.log('Registration result:', result); // Debug log

      // Handle different response statuses regardless of HTTP status code
      if (result.status === 'success') {
        setNotification({ 
          type: 'success', 
          message: result.message || 'All logos registered successfully!',
          details: result
        });
        
        // Clear form on complete success
        setFormData({
          companyName: '',
          websiteURL: '',
          metadata: '',
          logoFiles: [],
        });
        setPreviewURLs([]);
      } else if (result.status === 'partial') {
        setNotification({
          type: 'warning',
          message: result.message || 'Some logos registered successfully, some failed.',
          details: result,
          partialSuccess: true
        });
      } else if (result.status === 'failed') {
        setNotification({
          type: 'error',
          message: result.message || 'None of the logos could be registered',
          details: result
        });
      } else if (!response.ok) {
        // Handle similarity check failures
        if (result.failed_files && result.failed_files.length > 0) {
          const similarityFailures = result.failed_files.filter(f => f.stage === 'similarity_check');
          if (similarityFailures.length > 0) {
            setNotification({
              type: 'warning',
              message: 'Some logos could not be registered due to similarity.',
              details: result,
              similarityFailures
            });
          } else {
            setNotification({
              type: 'error',
              message: result.message || 'Registration failed',
              details: result
            });
          }
        } else {
          setNotification({
            type: 'error',
            message: result.message || 'Something went wrong'
          });
        }
      } else {
        setNotification({
          type: 'error',
          message: result.message || 'Registration failed',
          details: result
        });
      }
    } catch (error) {
      console.error('Error sending data:', error);
      setNotification({ type: 'error', message: 'Network error, please try again later.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheckBulkSimilarity = async () => {
    if (!formData.logoFiles.length) {
      setNotification({ type: 'error', message: 'Please upload at least one SVG logo to check.' });
      return;
    }

    setIsCheckingBulkSimilarity(true);
    setNotification(null);

    try {
      const data = new FormData();
      formData.logoFiles.forEach((file) => {
        data.append('logos', file);
      });

      const response = await fetch('http://localhost:5000/api/check-bulk-similarity', {
        method: 'POST',
        body: data,
      });

      const result = await response.json();

      if (response.ok) {
        if (result.status === 'all_can_register') {
          setNotification({
            type: 'success',
            message: 'All logos can be registered!',
            details: result
          });
        } else if (result.status === 'partial') {
          setNotification({
            type: 'warning',
            message: `Some logos can be registered, some cannot.`,
            details: result,
            bulkSimilarityCheck: true
          });
        } else {
          setNotification({
            type: 'warning',
            message: `None of the logos can be registered.`,
            details: result,
            bulkSimilarityCheck: true
          });
        }
      } else {
        setNotification({
          type: 'error',
          message: result.error || 'Failed to check bulk similarity'
        });
      }
    } catch (error) {
      console.error('Error checking bulk similarity:', error);
      setNotification({ type: 'error', message: 'Network error while checking bulk similarity.' });
    } finally {
      setIsCheckingBulkSimilarity(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const validSVGs = selectedFiles.filter(file => file.type === 'image/svg+xml');

    if (validSVGs.length !== selectedFiles.length) {
      alert('Only SVG files are allowed.');
      return;
    }

    const newPreviews = validSVGs.map(file => ({
      file,
      url: URL.createObjectURL(file),
    }));

    setFormData(prev => ({
      ...prev,
      logoFiles: [...prev.logoFiles, ...validSVGs],
    }));

    setPreviewURLs(prev => [...prev, ...newPreviews]);
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;

    if (type === 'checkbox') {
      setAllowMultiple(checked);
      // Clear all on toggle
      setFormData(prev => ({ ...prev, logoFiles: [] }));
      setPreviewURLs([]);
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleRemoveImage = (indexToRemove) => {
    // Remove the corresponding preview and file from the arrays
    const updatedPreviews = previewURLs.filter((_, idx) => idx !== indexToRemove);
    const updatedFiles = formData.logoFiles.filter((_, idx) => idx !== indexToRemove);

    // Update the state with the new arrays
    setPreviewURLs(updatedPreviews);
    setFormData(prev => ({
      ...prev,
      logoFiles: updatedFiles, // Remove the file from the logoFiles array
    }));
  };

  return (
    <div className="register-container">
      <h2 className="register-title">Register Your Logo</h2>
      <p className="register-desc">Upload and register your brand's logos in the domain name system.</p>

      <form className="register-form" onSubmit={handleSubmit}>
        <label>
          Company Name:
          <input
            type="text"
            name="companyName"
            value={formData.companyName}
            onChange={handleInputChange}
            required
          />
        </label>

        <label>
          Website URL:
          <input
            type="url"
            name="websiteURL"
            value={formData.websiteURL}
            onChange={handleInputChange}
            required
          />
        </label>

        <label>
          Metadata / Description:
          <textarea
            name="metadata"
            value={formData.metadata}
            onChange={handleInputChange}
            rows="4"
          />
        </label>

        <div className="checkbox-label">
          <input
            type="checkbox"
            checked={allowMultiple}
            onChange={handleInputChange}
          />
          <span>Upload different versions of the SVG logo</span>
        </div>

        <label>
          Upload Logo{allowMultiple ? 's' : ''} (SVG Only):
          <input
            type="file"
            name="logoFile"
            accept=".svg"
            multiple={allowMultiple}
            onChange={handleFileChange}
            required
          />
        </label>

        {previewURLs.length > 0 && (
          <div className="preview-box">
            <h4>Logo Preview{previewURLs.length > 1 ? 's' : ''}:</h4>
            <div className="preview-grid">
              {previewURLs.map((item, idx) => (
                <div key={idx} className="preview-wrapper">
                  <img
                    src={item.url}
                    alt={`Preview ${idx + 1}`}
                    className="logo-preview"
                  />
                  <button
                    type="button"
                    className="remove-btn"
                    onClick={() => handleRemoveImage(idx)}
                  >
                    ✖
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="button-group">
          <button 
            type="button" 
            className="check-bulk-btn" 
            onClick={handleCheckBulkSimilarity}
            disabled={isCheckingBulkSimilarity || !formData.logoFiles.length}
          >
            {isCheckingBulkSimilarity ? 'Checking...' : 'Check Similarity'}
          </button>
          <button type="submit" className="submit-btn" disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Submit'}
          </button>
        </div>
      </form>

      {/* Notification Display */}
      {notification && (
        <div className={`notification ${notification.type}`}>
          <div className="notification-header">
            <span className="notification-icon">
              {notification.type === 'success' && '✅'}
              {notification.type === 'error' && '❌'}
              {notification.type === 'warning' && '⚠️'}
            </span>
            <span className="notification-message">{notification.message}</span>
            <button 
              className="notification-close"
              onClick={() => setNotification(null)}
            >
              ×
            </button>
          </div>
          
          {/* Show similarity details for warning notifications */}
          {notification.type === 'warning' && (notification.similarityFailures || notification.bulkSimilarityCheck) && (
            <div className="similarity-details">
              <h4>Similar Logos Found:</h4>
              {notification.similarityFailures ? (
                notification.similarityFailures.map((failure, index) => (
                  <div key={index} className="similarity-item">
                    <h5>File: {failure.filename}</h5>
                    {failure.similar_logos && failure.similar_logos.length > 0 && (
                      <div className="similarity-table-container">
                                                <table className="similarity-table">
                          <thead>
                            <tr>
                              <th>Logo</th>
                              <th>Company</th>
                            </tr>
                          </thead>
                          <tbody>
                            {failure.similar_logos.map((logo, logoIndex) => (
                              <tr key={logoIndex}>
                                <td className="logo-image-cell">
                                  {logo.logo_image ? (
                                    <img 
                                      src={logo.logo_image} 
                                      alt={`Logo ${logoIndex + 1}`} 
                                      className="similarity-logo-image"
                                    />
                                  ) : (
                                    <div className="no-image-placeholder">No Image</div>
                                  )}
                                </td>
                                <td>{logo.company_name}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                ))
              ) : notification.bulkSimilarityCheck && (
                // Handle bulk similarity check results
                <div>
                  {notification.details.cannot_register && notification.details.cannot_register.map((file, fileIndex) => (
                    <div key={fileIndex} className="similarity-item">
                      <h5>File: {file.filename}</h5>
                      {file.similar_logos && file.similar_logos.length > 0 && (
                        <div className="similarity-table-container">
                          <table className="similarity-table">
                            <thead>
                              <tr>
                                <th>Logo</th>
                                <th>Company</th>
                              </tr>
                            </thead>
                            <tbody>
                              {file.similar_logos.map((logo, logoIndex) => (
                                <tr key={logoIndex}>
                                  <td className="logo-image-cell">
                                    {logo.logo_image ? (
                                      <img 
                                        src={logo.logo_image} 
                                        alt={`Logo ${logoIndex + 1}`} 
                                        className="similarity-logo-image"
                                      />
                                    ) : (
                                      <div className="no-image-placeholder">No Image</div>
                                    )}
                                  </td>
                                  <td>{logo.company_name}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Show success details */}
          {notification.type === 'success' && notification.details && (
            <div className="success-details">
              {notification.details.results ? (
                <p>All {notification.details.results.length} logo(s) registered successfully!</p>
              ) : notification.details.can_register ? (
                <p>All {notification.details.can_register.length} logo(s) can be registered!</p>
              ) : (
                <p>Operation completed successfully</p>
              )}
            </div>
          )}

          {/* Show partial success details */}
          {notification.type === 'warning' && notification.partialSuccess && notification.details && (
            <div className="partial-success-details">
              <p>✅ {notification.details.summary?.successful || 0} logo(s) registered successfully</p>
              <p>❌ {notification.details.summary?.failed || 0} logo(s) failed</p>
              {notification.details.summary?.similarity_failures > 0 && (
                <p>⚠️ {notification.details.summary.similarity_failures} failed due to similarity</p>
              )}
              
              {/* Show failed file names */}
              {notification.details.failed_files && notification.details.failed_files.length > 0 && (
                <div className="failed-files-list">
                  <h5>Failed Files:</h5>
                  <ul>
                    {notification.details.failed_files.map((failedFile, index) => (
                      <li key={index} className="failed-file-item">
                        <span className="failed-filename">{failedFile.filename}</span>
                        <span className="failed-reason"> - {failedFile.error}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RegisterLogo;



















// import React, { useState } from 'react';
// import './RegisterLogo.css';

// const RegisterLogo = () => {
//   const [formData, setFormData] = useState({
//     companyName: '',
//     websiteURL: '',
//     metadata: '',
//     logoFile: null,
//   });

//   const [previewURL, setPreviewURL] = useState(''); // New state for preview

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (!formData.logoFile || formData.logoFile.type !== 'image/svg+xml') {
//       alert('Please upload a valid SVG file.');
//       return;
//     }

//     const data = new FormData();
//     data.append('companyName', formData.companyName);
//     data.append('websiteURL', formData.websiteURL);
//     data.append('metadata', formData.metadata);
//     data.append('logo', formData.logoFile);

//     try {
//       const response = await fetch('http://localhost:5000/api/register-logo', {
//         method: 'POST',
//         body: data,
//       });

//       if (response.ok) {
//         const result = await response.json();
//         alert('Logo registered successfully!');
//         console.log('Server response:', result);
//       } else {
//         const errorData = await response.json();
//         alert(`Error: ${errorData.message || 'Something went wrong'}`);
//       }
//     } catch (error) {
//       console.error('Error sending data:', error);
//       alert('Network error, please try again later.');
//     }
//   };

//   const handleChange = (e) => {
//     const { name, value, files } = e.target;
//     if (name === 'logoFile') {
//       const file = files[0];
//       if (file && file.type === 'image/svg+xml') {
//         setFormData({ ...formData, logoFile: file });
//         setPreviewURL(URL.createObjectURL(file)); // Set preview URL
//       } else {
//         alert('Please upload a valid SVG file.');
//         setFormData({ ...formData, logoFile: null });
//         setPreviewURL('');
//       }
//     } else {
//       setFormData({ ...formData, [name]: value });
//     }
//   };

//   return (
//     <div className="register-container">
//       <h2 className="register-title">Register Your Logo</h2>
//       <p className="register-desc">Upload and register your brand's logos in the domain name system.</p>

//       <form className="register-form" onSubmit={handleSubmit}>
//         <label>
//           Company Name:
//           <input
//             type="text"
//             name="companyName"
//             value={formData.companyName}
//             onChange={handleChange}
//             required
//           />
//         </label>

//         <label>
//           Website URL:
//           <input
//             type="url"
//             name="websiteURL"
//             value={formData.websiteURL}
//             onChange={handleChange}
//             required
//           />
//         </label>

//         <label>
//           Metadata / Description:
//           <textarea
//             name="metadata"
//             value={formData.metadata}
//             onChange={handleChange}
//             rows="4"
//           />
//         </label>

//         <label>
//           Upload Logo (SVG Only):
//           <input
//             type="file"
//             name="logoFile"
//             accept=".svg"
//             onChange={handleChange}
//             required
//           />
//         </label>

//         {/* Preview */}
//         {previewURL && (
//           <div className="preview-box">
//             <img src={previewURL} alt="Logo Preview" className="logo-preview" />
//           </div>
//         )}

//         <button type="submit" className="submit-btn">Submit</button>
//       </form>
//     </div>
//   );
// };

// export default RegisterLogo;
