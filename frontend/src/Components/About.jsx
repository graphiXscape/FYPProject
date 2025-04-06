import React from 'react';
import './About.css'; // Make sure to import the CSS file

const About = () => {
  return (
    <div className="about-container">
      <h2 className="about-title">Our Research Project</h2>
      <p className="about-subtitle">Comparative Analysis of Brand Visual Identification</p>
      <p className="about-content">
        Brand visuals such as logos play a major role in modern digital branding, allowing companies to establish a strong visual presence. As brand visuals are central to the identity of a brand, an efficient method for recognising brand visuals in documents, posters, and other media is essential in applications such as brand recognition, content tagging, and looking up a visual in a database. Although image recognition is a mature field, less work has been done on recognising 2D line and shape images, especially where the visuals are relatively simple, but high accuracy is desired. This research consists of a comparative evaluation of brand visual recognition methods for visuals encoded as vector graphics. Vector graphics define an image as a set of points, lines, and curves to create scalable, resolution-independent visuals. They are suitable for images which comprise 2-D figures comprising lines and shapes. Our study examined algorithm-based approaches for brand visual comparison, including Hausdorff distance, Procrustes distance, Earth Mover's distance, and centroid distance and angle feature (CDAF) matching, along with deep learning approaches such as DeepSVG. Five of these methods were evaluated using a set of query images with varying levels of impairments on a database of brand visuals. We show that Procrustes Analysis and DeepSVG provide good matching even when the query image is significantly distorted from the reference image in the database. The correct image received the no. 1 ranking in over 90% of the tests. The best results from both the algorithmic methods and deep learning methods show similar performance, which indicates that deep learning methods are not necessarily superior to algorithmic methods.
      </p>
    </div>
  );
};

export default About;
