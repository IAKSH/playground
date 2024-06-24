// pages/_app.js
import React from 'react';
import { NextUIProvider } from '@nextui-org/react';

function MyApp({ Component, pageProps }) {
    return (
        <NextUIProvider>
        <div style={{
            backgroundImage: 'url(/images/pattern.svg)',
            //backgroundSize: 'cover',
            height: '98vh',
        }}>
            <div style={{
                backgroundColor: 'rgba(255, 255, 255, 0.0)',
                //backdropFilter: 'blur(0.5px)',
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: 0,
                right: 0,
                padding: '1em',
            }}></div>
            <Component {...pageProps} />
        </div>
        </NextUIProvider>
    )
}

export default MyApp;