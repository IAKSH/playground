import React, { useState, useEffect } from 'react';
import mqtt from 'mqtt';
import { AppBar, Toolbar, Typography, Container, Slider, Box, Paper, Switch, FormControlLabel, Grid } from '@mui/material';

const App = () => {
  const [client, setClient] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [reconnects, setReconnects] = useState(0);
  const [motorOffset, setMotorOffset] = useState(0);
  const [kp, setKp] = useState(0);
  const [ki, setKi] = useState(0);
  const [kd, setKd] = useState(0);
  const [roll, setRoll] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [yaw, setYaw] = useState(0);
  const [emergencyStop, setEmergencyStop] = useState(false);
  const [gyroEuler, setGyroEuler] = useState([0, 0, 0]);
  const [gyroAccel, setGyroAccel] = useState([0, 0, 0]);
  const [gyroTemperature, setGyroTemperature] = useState(0);
  const [barometerPressure, setBarometerPressure] = useState(0);
  const [barometerTemperature, setBarometerTemperature] = useState(0);
  const [barometerAltitude, setBarometerAltitude] = useState(0);
  const [motorDuty, setMotorDuty] = useState([0, 0, 0, 0]);

  useEffect(() => {
    const mqttClient = mqtt.connect('ws://192.168.31.180:8083');

    mqttClient.on('connect', () => {
      setStatus('Connected');
      const topics = [
        '/drone/control/motor_offset', '/drone/control/kp', '/drone/control/ki', 
        '/drone/control/kd', '/drone/control/euler/roll', '/drone/control/euler/pitch',
        '/drone/control/euler/yaw', '/drone/control/emergency_stop', '/drone/gryo/euler',
        '/drone/gryo/accel', '/drone/gryo/temperature', '/drone/barometer/pressure',
        '/drone/barometer/temperature', '/drone/barometer/altitude', '/drone/motor/duty'
      ];
      topics.forEach(topic => mqttClient.subscribe(topic));
    });

    mqttClient.on('reconnect', () => {
      setReconnects(prev => prev + 1);
      setStatus('Reconnecting...');
    });

    mqttClient.on('disconnect', () => {
      setStatus('Disconnected');
    });

    mqttClient.on('message', (topic, message) => {
      const value = parseFloat(message.toString());
      switch (topic) {
        case '/drone/control/motor_offset':
          setMotorOffset(value);
          break;
        case '/drone/control/kp':
          setKp(value);
          break;
        case '/drone/control/ki':
          setKi(value);
          break;
        case '/drone/control/kd':
          setKd(value);
          break;
        case '/drone/control/euler/roll':
          setRoll(value);
          break;
        case '/drone/control/euler/pitch':
          setPitch(value);
          break;
        case '/drone/control/euler/yaw':
          setYaw(value);
          break;
        case '/drone/control/emergency_stop':
          setEmergencyStop(value === 1);
          break;
        case '/drone/gryo/euler':
          setGyroEuler(message.toString().split(',').map(Number));
          break;
        case '/drone/gryo/accel':
          setGyroAccel(message.toString().split(',').map(Number));
          break;
        case '/drone/gryo/temperature':
          setGyroTemperature(value);
          break;
        case '/drone/barometer/pressure':
          setBarometerPressure(value);
          break;
        case '/drone/barometer/temperature':
          setBarometerTemperature(value);
          break;
        case '/drone/barometer/altitude':
          setBarometerAltitude(value);
          break;
        case '/drone/motor/duty':
          setMotorDuty(message.toString().split(',').map(Number));
          break;
        default:
          break;
      }
    });

    setClient(mqttClient);

    return () => {
      mqttClient.end();
    };
  }, []);

  const handleSliderChange = (topic, value) => {
    if (client && client.connected) {
      client.publish(topic, value.toString());
    }
  };

  const handleSwitchChange = (event) => {
    const value = event.target.checked ? 1 : 0;
    setEmergencyStop(event.target.checked);
    if (client && client.connected) {
      client.publish('/drone/control/emergency_stop', value.toString());
    }
  };

  return (
    <Container>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">
            MQTT Drone Control
          </Typography>
        </Toolbar>
      </AppBar>
      <Paper elevation={3} style={{ padding: 20, marginTop: 20 }}>
        <Box mb={2}>
          <Typography>Status: {status}</Typography>
        </Box>
        <Box mb={2}>
          <Typography>Reconnect Attempts: {reconnects}</Typography>
        </Box>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Motor Offset</Typography>
            <Slider
              value={motorOffset}
              min={0}
              max={8192}
              onChange={(e, value) => handleSliderChange('/drone/control/motor_offset', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Kp</Typography>
            <Slider
              value={kp}
              min={0}
              max={50}
              step={0.1}
              onChange={(e, value) => handleSliderChange('/drone/control/kp', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Ki</Typography>
            <Slider
              value={ki}
              min={0}
              max={50}
              step={0.1}
              onChange={(e, value) => handleSliderChange('/drone/control/ki', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Kd</Typography>
            <Slider
              value={kd}
              min={0}
              max={50}
              step={0.1}
              onChange={(e, value) => handleSliderChange('/drone/control/kd', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Euler Roll</Typography>
            <Slider
              value={roll}
              min={-180}
              max={180}
              onChange={(e, value) => handleSliderChange('/drone/control/euler/roll', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Euler Pitch</Typography>
            <Slider
              value={pitch}
              min={-180}
              max={180}
              onChange={(e, value) => handleSliderChange('/drone/control/euler/pitch', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography gutterBottom>Euler Yaw</Typography>
            <Slider
              value={yaw}
              min={-180}
              max={180}
              onChange={(e, value) => handleSliderChange('/drone/control/euler/yaw', value)}
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={emergencyStop}
                  onChange={handleSwitchChange}
                  color="primary"
                />
              }
              label={emergencyStop ? 'Emergency Stop Activated' : 'Emergency Stop Deactivated'}
            />
          </Grid>
        </Grid>
        <Box mt={4}>
          <Typography variant="h6">Drone Data</Typography>
          <Typography>Gyro Euler: {gyroEuler.join(', ')}</Typography>
          <Typography>Gyro Acceleration: {gyroAccel.join(', ')}</Typography>
          <Typography>Gyro Temperature: {gyroTemperature}°C</Typography>
          <Typography>Barometer Pressure: {barometerPressure} hPa</Typography>
          <Typography>Barometer Temperature: {barometerTemperature}°C</Typography>
          <Typography>Barometer Altitude: {barometerAltitude} m</Typography>
          <Typography>Motor Duty: {motorDuty.join(', ')}</Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default App;
