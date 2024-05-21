<template>
    <div id="app">
      <sidebar @change-page="changePage"></sidebar>
      <div id="content">
        <component :is="currentPage"></component>
      </div>
    </div>
  </template>
  
  <script>
  import Sidebar from './components/Sidebar.vue'
  import Statistics from './components/Statistics.vue'
  import Sales from './components/Sales.vue'
  import Purchase from './components/Purchase.vue'
  import Member from './components/Member.vue'
  import Staff from './components/Staff.vue'
  
  export default {
    name: 'App',
    components: {
      Sidebar,
      Statistics,
      Sales,
      Purchase,
      Member,
      Staff
    },
    data() {
      return {
        currentPage: 'Statistics'
      }
    },
    methods: {
      changePage(page) {
        this.currentPage = page;
      }
    }
  }
  </script>
  
  <style>
  #app {
    display: flex;
  }
  #content {
    margin-left: 200px;
    padding: 20px;
  }
  </style>
  