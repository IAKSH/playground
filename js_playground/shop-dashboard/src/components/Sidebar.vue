<template>
    <div id="sidebar">
      <ul>
        <li @click="changePage('Statistics')">统计</li>
        <li @click="changePage('Sales')">售出</li>
        <li @click="changePage('Purchase')">进货</li>
        <li @click="changePage('Member')">会员</li>
        <li @click="changePage('Staff')">员工</li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    name: 'Sidebar',
    methods: {
      changePage(page) {
        this.$emit('change-page', page);
      }
    }
  }
  </script>
  
  <style>
  #sidebar {
    width: 200px;
    height: 100vh;
    background-color: #f8f9fa;
    padding: 20px;
  }
  #sidebar ul {
    list-style-type: none;
    padding: 0;
  }
  #sidebar ul li {
    cursor: pointer;
    margin-bottom: 20px;
  }
  </style>
  