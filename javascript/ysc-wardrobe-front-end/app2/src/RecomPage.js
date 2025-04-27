import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ItemCard from './ItemCard';
import './RecomPage.css';

function Recommendations() {
  const [items, setItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  // Function to fetch items from the API and randomly select 8 items
  const fetchItems = async () => {
    try {
      const response = await axios.get('/api/home');
      if (response.data.code === 1) {
        const allItems = response.data.data;
        const shuffledItems = allItems.sort(() => 0.5 - Math.random());
        const selectedItems = shuffledItems.slice(0, 8);
        setItems(selectedItems);
      } else {
        console.error('Failed to fetch items:', response.data.msg);
      }
    } catch (error) {
      console.error('Error fetching items:', error);
    }
  };

  useEffect(() => {
    // Fetch items immediately and then every 30 seconds
    fetchItems();
    const interval = setInterval(fetchItems, 30000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

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

    const loadingTime = Math.floor(Math.random() * 1000) + 500;

    setTimeout(() => {
      setIsLoading(false);
      setItems(items.filter(item => item !== selectedItem));
      setSelectedItem(null);
      setIsSuccess(true);

      handleClose();

      setTimeout(() => {
        setIsSuccess(false);
      }, 1000);
    }, loadingTime);

    handleClose();
  };

  return (
    <div className="recom-page">
      <div className="content-container">
        <div className="items-container">
          {items.map((item, index) => (
            <ItemCard key={index} item={item} onClick={() => handleClick(item)} />
          ))}
        </div>
      </div>

      {selectedItem && (
        <div className="modal-overlay" onClick={handleClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <img src={selectedItem.image} alt={selectedItem.name} className="modal-image" />
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

      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-content">连接中...</div>
        </div>
      )}

      {isSuccess && (
        <div className="success-dialog">
          预定成功
        </div>
      )}

    </div>
  );
}

export default Recommendations;
