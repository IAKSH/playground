// components/Sidebar.jsx
import React from 'react';

const Sidebar = ({ selectedItem, setSelectedItem, links }) => (
  <div style={{
    width: '200px',
    height: '98vh',
    backgroundImage: 'url(https://images.pexels.com/photos/1103970/pexels-photo-1103970.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)',
    backgroundSize: 'cover',
    //borderRadius: '5x',
    boxShadow: '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
    position: 'relative',
  }}>
    <div style={{
      backgroundColor: 'rgba(255, 255, 255, 0.75)',
      backdropFilter: 'blur(2px)',
      position: 'absolute',
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      padding: '1em',
    }}>
      <h2>Network</h2>
      <ul style={{ listStyleType: 'none', padding: 0 }}>
        {Object.keys(links).map(link => (
          <li key={link} style={getLinkStyle(selectedItem === link)} onClick={() => setSelectedItem(link)}>
            {link}
          </li>
        ))}
      </ul>
    </div>
  </div>
)

const getLinkStyle = (isSelected) => ({
  padding: '1em',
  borderRadius: '5px',
  backgroundColor: isSelected ? 'blue' : 'transparent',
  color: isSelected ? 'white' : 'black',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  ':hover': {
    backgroundColor: 'lightblue',
  },
  ':active': {
    backgroundColor: 'darkblue',
  },
});

export default Sidebar;
