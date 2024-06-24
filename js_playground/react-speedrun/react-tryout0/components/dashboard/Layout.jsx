// Layout.jsx
import React from 'react';
import Card from './RotateCard';
import { Button } from "@nextui-org/button";

const Layout = () => (
    <div>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <Card>你好</Card>
            <Card>你好</Card>
            <Card>你好</Card>
            <Card>你好</Card>
        </div>
        <Button>Press me</Button>
    </div>
);

export default Layout;
