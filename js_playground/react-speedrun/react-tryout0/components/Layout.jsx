// components/Layout.jsx
import React from 'react';
import Sidebar from './Sidebar';

const Layout = ({ children, selectedItem, setSelectedItem, links}) => (
    <div style={{ display: 'flex' }}>
        <Sidebar selectedItem={selectedItem} setSelectedItem={setSelectedItem} links={links} />
        <main style={{ flex: 1, marginLeft: '20px' }}>{children}</main>
    </div>
);

export default Layout;
