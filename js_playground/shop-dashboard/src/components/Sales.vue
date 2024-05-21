<template>
    <div>
      <h1>售出</h1>
      <!-- 这里应该是从服务器获取的Sales数据 -->
      <!-- 这里应该是一个表格，显示Sales数据 -->
      <!-- 这里应该是一些按钮和输入框，用于向后端添加一条Sales记录 -->
    </div>
  </template>
  
  <script>
  export default {
    name: 'Sales',
    data() {
      return {
        // 这里应该是从服务器获取的Sales数据
      }
    },
    created() {
      // 在这里发送请求到服务器获取Sales数据
    },
    methods: {
      updateSales(sales) {
        // 在这里发送请求到服务器更新Sales数据
      },
      insertSales(sales) {
        // 在这里发送请求到服务器插入Sales数据
      },
      deleteSales(sales) {
        // 在这里发送请求到服务器删除Sales数据
      }
    }
  }
  </script>
  