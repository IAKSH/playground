<template>
  <div>
    <h1>进货</h1>
    <table>
      <tr v-for="purchase in purchases" :key="purchase.purchaseID">
        <td>{{ purchase.productID }}</td>
        <td>{{ purchase.purchaseTime }}</td>
        <td>{{ purchase.purchaseUnitPrice }}</td>
        <td>{{ purchase.purchaseQuantity }}</td>
        <td>{{ purchase.sourceID }}</td>
        <td>
          <button @click="updatePurchase(purchase)">修改</button>
          <button @click="deletePurchase(purchase)">删除</button>
        </td>
      </tr>
    </table>
    <div>
      <input v-model="newPurchase.productID" placeholder="产品ID" />
      <input v-model="newPurchase.purchaseTime" placeholder="进货时间" />
      <input v-model="newPurchase.purchaseUnitPrice" placeholder="进货单价" />
      <input v-model="newPurchase.purchaseQuantity" placeholder="进货数量" />
      <input v-model="newPurchase.sourceID" placeholder="来源ID" />
      <button @click="insertPurchase(newPurchase)">添加进货记录</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'Purchase',
  data() {
    return {
      purchases: [],
      newPurchase: {
        productID: '',
        purchaseTime: '',
        purchaseUnitPrice: '',
        purchaseQuantity: '',
        sourceID: ''
      }
    }
  },
  created() {
    axios.get('/api/purchase/all')
      .then(response => {
        this.purchases = response.data;
      });
  },
  methods: {
    updatePurchase(purchase) {
      axios.post('/api/purchase/update', purchase)
        .then(response => {
          // 更新成功
        });
    },
    insertPurchase(purchase) {
      axios.post('/api/purchase/insert', purchase)
        .then(response => {
          // 插入成功
          this.purchases.push(purchase);
        });
    },
    deletePurchase(purchase) {
      axios.get(`/api/purchase/delete/${purchase.purchaseID}`)
        .then(response => {
          // 删除成功
          const index = this.purchases.indexOf(purchase);
          this.purchases.splice(index, 1);
        });
    }
  }
}
</script>
