<template>
    <div>
      <h1>员工</h1>
      <!-- 这里应该是从服务器获取的Staff数据 -->
      <!-- 这里应该是一个表格，显示Staff数据 -->
      <!-- 这里应该是一些按钮和输入框，用于向后端添加一条Staff记录 -->
    </div>
  </template>
  
  <script>
  export default {
    name: 'Staff',
    data() {
      return {
        // 这里应该是从服务器获取的Staff数据
      }
    },
    created() {
      // 在这里发送请求到服务器获取Staff数据
    },
    methods: {
      updateStaff(staff) {
        // 在这里发送请求到服务器更新Staff数据
      },
      insertStaff(staff) {
        // 在这里发送请求到服务器插入Staff数据
      },
      deleteStaff(staff) {
        // 在这里发送请求到服务器删除Staff数据
      }
    }
  }
  </script>
  