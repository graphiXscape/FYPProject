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

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.logoFiles.length) {
      alert('Please upload at least one SVG logo.');
      return;
    }

    const invalidFiles = formData.logoFiles.filter(file => file.type !== 'image/svg+xml');
    if (invalidFiles.length > 0) {
      alert('Please make sure all uploaded files are SVG images.');
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

      if (response.ok) {
        const result = await response.json();
        alert('Logo(s) registered successfully!');
        console.log('Server response:', result);
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.message || 'Something went wrong'}`);
      }
    } catch (error) {
      console.error('Error sending data:', error);
      alert('Network error, please try again later.');
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

        <button type="submit" className="submit-btn">Submit</button>
      </form>
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
