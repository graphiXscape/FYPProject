import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css'; // ⬅️ Import your CSS file

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <h1 className="home-title">Welcome to the Brand Visual Registration and Lookup System</h1>
      
      <p className="home-description">
      Our system offers a streamlined experience for managing your brand visuals, focusing on Scalable Vector Graphics (SVG) for vector images. Easily register and store your brand’s logo and identity in SVG format, while enabling quick and efficient lookup of existing brand profiles. Designed with simplicity, speed, and clarity, our system ensures seamless access and management of your brand’s visual assets.
      </p>

      <div className="home-buttons">
        <button onClick={() => navigate('/register')} className="btn btn-register">
          Register
        </button>
        <button onClick={() => navigate('/lookup')} className="btn btn-lookup">
          Look Up
        </button>
      </div>
    </div>
  );
};

export default Home;
