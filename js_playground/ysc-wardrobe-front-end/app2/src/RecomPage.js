import React, { useState } from 'react';
import ItemCard from './ItemCard';
import './RecomPage.css';

function Recommendations() {
  const [items, setItems] = useState([
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    },{
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    }
    // 添加更多项...
  ]);

  const [selectedItem, setSelectedItem] = useState(null);

  const handleClick = (item) => {
    setSelectedItem(item);
  };

  const handleNewRecom = () => {
    setSelectedItem(null);
  };

  const handleClose = () => {
    setSelectedItem(null);
  };

  const handleReserve = () => {
    console.log('从推荐预定', selectedItem);
    handleClose();
  };

  return (
    <div className="recom-page">
      <div className="content-container">
        <div className="items-container">
          {items.map((item, index) => (
            <ItemCard key={index} item={item} onClick={handleClick} />
          ))}
        </div>
        <div className="button-container">
          <button className="more-recom-button" onClick={handleNewRecom}>更多推荐</button>
        </div>
      </div>

      {selectedItem && (
        <div className="modal-overlay" onClick={handleClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <img src={selectedItem.imageUrl} alt={selectedItem.name} className="modal-image" />
            <div className="modal-info">
              <p><span>名称:</span> <span>{selectedItem.name}</span></p>
              <p><span>大小:</span> <span>{selectedItem.size}</span></p>
              <p><span>价格:</span> <span>{selectedItem.price}</span></p>
            </div>
            <div className="modal-actions">
              <button className="reserve-button" onClick={handleReserve}>前往预定</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Recommendations;
