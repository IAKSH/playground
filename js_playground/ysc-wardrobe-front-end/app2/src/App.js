import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Navbar from './Navbar';
import Home from './Home';
import ReturnPage from './ReturnPage';
import ReservePage from './ReservePage';
import RecomPage from './RecomPage';
import MapPage from './MapPage';
import TryOutPage from './TryOutPage'
import './App.css';

function App() {
    const userID = "lain"; // 示例用户ID

    return (
        <Router>
            <div className="App">
                <Navbar />
                <div className="content">
                  <Routes>
                      <Route path="/" exact element={<Home />} />
                      <Route path="/return" element={<ReturnPage />} />
                      <Route path="/reserve" element={<ReservePage />} />
                      <Route path="/recommendations" element={<RecomPage />} />
                      <Route path="/map" element={<MapPage />} />
                      <Route path="/tryout" element={<TryOutPage />} />
                  </Routes>
                </div>
                <div className="footer">
                    <div className="user-id">用户ID: {userID}</div>
                    <Link to="/map">
                        <img src="static/ysc_map.png" alt="logo" className="Map-logo" />
                    </Link>
                </div>
            </div>
        </Router>
    );
}

export default App;
