import React from 'react';
import './Navbar.css'; // Make sure to create this file

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo">GraphiXscape</div>
      <ul className="nav-links">
        <li><a href="/">Home</a></li>
        <li><a href="/about">About</a></li>
        <li><a href="/register">Register Logo</a></li>
        <li><a href="/lookup">Lookup Logo</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;
