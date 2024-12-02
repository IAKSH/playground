import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ItemCard from './ItemCard';
import './ReservePage.css';

function ReservePage() {
  const [items, setItems] = useState([
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: '2024-01-01',
      returnDate: '2024-01-10',
      image: 'static/b1.jpg'
    },
    {
        name: '汉服B',
        size: 'L',
        price: '￥200',
        borrowedDate: '2024-01-01',
        returnDate: '2024-01-10',
        image: 'static/c2.jpg'
      },
      {
        name: '汉服C',
        size: 'L',
        price: '￥200',
        borrowedDate: '2024-01-01',
        returnDate: '2024-01-10',
        image: 'static/a2.jpg'
      }
    // 添加更多项...
  ]);

  const [selectedItem, setSelectedItem] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  

  /*
  useEffect(() => {
    const fetchItems = async () => {
      try {
        const response = await axios.get('/api/home');
        if (response.data.code === 1) {
          setItems(response.data.data);
        } else {
          console.error('Failed to fetch items:', response.data.msg);
        }
      } catch (error) {
        console.error('Error fetching items:', error);
      }
    };

    // Fetch items immediately and then every 30 second
    fetchItems();
    const interval = setInterval(fetchItems, 30000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);
  */

  const handleClick = (item) => {
    setSelectedItem(item);
  };

  const handleClose = () => {
    setSelectedItem(null);
  };

  const handleTakeOut = () => {
    console.log('立即取衣', selectedItem);

    setIsLoading(true);

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
  };

  return (
    <div className="return-page">
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
              <p><span>店铺:</span> <span>{selectedItem.shopName}</span></p>
              <p><span>价格:</span> <span>{selectedItem.price}</span></p>
            </div>
            <div className="modal-actions">
              <button className="return-button" onClick={handleTakeOut}>立即取衣</button>
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
          取衣成功
        </div>
      )}
    </div>
  );
}

export default ReservePage;
