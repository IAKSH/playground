import React from 'react';
import { Link } from 'react-router-dom';
import './MainButtons.css';

function MainButtons() {
  return (
    <div className="buttons-container">
      <Link to="/return" className="button">
        <img src="static/ysc_arrow.png" alt="logo" className="button-logo" />
        <div className="button-text">
          <i className="fas fa-arrow-down"></i>
          归还<br />服饰
        </div>
      </Link>
      <Link to="/reserve" className="button">
        <img src="static/ysc_container.png" alt="logo" className="button-logo" />
        <div className="button-text">
          <i className="fas fa-calendar-alt"></i>
          预定<br />取衣
        </div>
      </Link>
      <Link to="/recommendations" className="button">
        <img src="static/ysc_search.png" alt="logo" className="button-logo" />
        <div className="button-text">
          <i className="fas fa-question-circle"></i>
          智能<br />推荐
        </div>
      </Link>
    </div>
  );
}

export default MainButtons;
