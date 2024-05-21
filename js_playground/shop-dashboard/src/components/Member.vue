<template>
  <div>
    <h1>会员</h1>
    <table>
      <tr v-for="member in members" :key="member.memberID">
        <td>{{ member.name }}</td>
        <td>{{ member.membershipStartDate }}</td>
        <td>{{ member.membershipEndDate }}</td>
        <td>
          <button @click="updateMember(member)">修改</button>
          <button @click="deleteMember(member)">删除</button>
        </td>
      </tr>
    </table>
    <div>
      <input v-model="newMember.name" placeholder="姓名" />
      <input v-model="newMember.membershipStartDate" placeholder="会员开始日期" />
      <input v-model="newMember.membershipEndDate" placeholder="会员结束日期" />
      <button @click="insertMember(newMember)">添加会员</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'Member',
  data() {
    return {
      members: [],
      newMember: {
        name: '',
        membershipStartDate: '',
        membershipEndDate: ''
      }
    }
  },
  created() {
    axios.get('/api/member/all')
      .then(response => {
        this.members = response.data;
      });
  },
  methods: {
    updateMember(member) {
      axios.post('/api/member/update', member)
        .then(response => {
          // 更新成功
        });
    },
    insertMember(member) {
      axios.post('/api/member/insert', member)
        .then(response => {
          // 插入成功
          this.members.push(member);
        });
    },
    deleteMember(member) {
      axios.get(`/api/member/delete/${member.memberID}`)
        .then(response => {
          // 删除成功
          const index = this.members.indexOf(member);
          this.members.splice(index, 1);
        });
    }
  }
}
</script>
