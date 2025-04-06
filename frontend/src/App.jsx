import './App.css';
import { Routes, Route } from 'react-router-dom';
import Home from './Components/Home';
import Navbar from './Components/NavBar';
import RegisterLogo from './Components/RegisterLogo'; 
import Lookup from './Components/LookupLogos';
import About from './Components/About';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/register" element={<RegisterLogo />} />
        <Route path="/lookup" element = {<Lookup />}/>
        <Route path="/about" element = {< About/>}/>
      </Routes>
    </div>
  );
}

export default App;
