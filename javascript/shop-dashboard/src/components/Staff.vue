<template>
  <div>
    <h1>员工</h1>
    <table>
      <tr v-for="staff in staffs" :key="staff.staffID">
        <td>{{ staff.name }}</td>
        <td>{{ staff.gender }}</td>
        <td>{{ staff.age }}</td>
        <td>{{ staff.monthlySalary }}</td>
        <td>
          <button @click="updateStaff(staff)">修改</button>
          <button @click="deleteStaff(staff)">删除</button>
        </td>
      </tr>
    </table>
    <div>
      <input v-model="newStaff.name" placeholder="姓名" />
      <input v-model="newStaff.gender" placeholder="性别" />
      <input v-model="newStaff.age" placeholder="年龄" />
      <input v-model="newStaff.monthlySalary" placeholder="月薪" />
      <button @click="insertStaff(newStaff)">添加员工记录</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'Staff',
  data() {
    return {
      staffs: [],
      newStaff: {
        name: '',
        gender: '',
        age: '',
        monthlySalary: ''
      }
    }
  },
  created() {
    axios.get('/api/staff/all')
      .then(response => {
        this.staffs = response.data;
      });
  },
  methods: {
    updateStaff(staff) {
      axios.post('/api/staff/update', staff, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
        .then(response => {
          // 更新成功
        });
    },
    insertStaff(staff) {
      axios.post('/api/staff/insert', staff, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
        .then(response => {
          // 插入成功
          this.staffs.push(staff);
        });
    },
    deleteStaff(staff) {
      axios.get(`/api/staff/delete/${staff.staffID}`)
        .then(response => {
          // 删除成功
          const index = this.staffs.indexOf(staff);
          this.staffs.splice(index, 1);
        });
    }
  }
}
</script>
