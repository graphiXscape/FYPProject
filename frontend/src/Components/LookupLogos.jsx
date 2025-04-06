import React, { useState } from 'react';
import './Lookup.css';

const Lookup = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false); // New loading state

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'image/svg+xml') {
      setSelectedFile(file);
      setPreviewURL(URL.createObjectURL(file));
    } else {
      alert('Please upload a valid SVG file.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select an SVG image to look up.');
      return;
    }

    const formData = new FormData();
    formData.append('logo', selectedFile);

    try {
      setLoading(true); // Show loading message
      setResults([]);   // Clear previous results while loading

      const res = await fetch('http://localhost:5000/api/lookup-logo', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      setResults(data.matches || []);
    } catch (error) {
      console.error('Error:', error);
      alert('Error during lookup.');
    } finally {
      setLoading(false); // Hide loading message
    }
  };

  return (
    <div className="lookup-container">
      <h2 className="lookup-title">Logo Lookup</h2>
      <p>Upload an SVG logo to find similar registered brands.</p>

      <form onSubmit={handleSubmit} className="lookup-form">
        <input type="file" accept=".svg" onChange={handleFileChange} />

        {previewURL && (
          <div className="preview-box">
            <h4>Preview:</h4>
            <img src={previewURL} alt="SVG Preview" />
          </div>
        )}

        <button type="submit" className="lookup-btn">Submit</button>
      </form>

      {/* Loading message */}
      {loading && <p className="loading-msg">Loading results...</p>}

      {/* Results */}
      {!loading && results.length > 0 && (
        <div className="results-section">
          <h3>Top Matches</h3>
          <div className="results-grid">
            {results.map((item, idx) => (
              <div key={idx} className="result-card">
                <img src={item.logoUrl} alt={`Match ${idx + 1}`} />
                <a href={item.companyUrl} target="_blank" rel="noopener noreferrer">
                  {item.companyUrl}
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Lookup;
