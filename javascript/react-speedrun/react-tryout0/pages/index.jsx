// pages/index.js
import React, { useState } from 'react';
import Layout from '../components/Layout';
import DashboardLayout from "../components/dashboard/Layout"

const HomePage = () => {
    const [selectedItem, setSelectedItem] = useState('链接1');

    const links = {
        'Dashboard': <DashboardLayout />,
        'Nodes': <h1>where you check your nodes</h1>,
        'Payments': <h1>where you pay</h1>,
        'User': <h1>where you edit your user info</h1>,
    }

    return (
        <Layout selectedItem={selectedItem} setSelectedItem={setSelectedItem} links={links}>
            {Object.keys(links).map(link => {
                if (selectedItem === link) {
                    return links[link];
                }
                return null;
            })}
        </Layout>
    );
};

export default HomePage;
