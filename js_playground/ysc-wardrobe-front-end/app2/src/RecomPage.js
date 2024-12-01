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
      imageUrl: 'static/a1.jpg'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'static/b2.jpg'
    },{
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'static/c2.jpg'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'static/c1.jpg'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'static/a3.jpg'
    },
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'static/b3.jpg'
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
