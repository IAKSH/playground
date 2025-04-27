<template>
    <div>
        <h1>售出</h1>
        <table>
            <tr v-for="sale in sales" :key="sale.salesID">
                <td>{{ sale.productID }}</td>
                <td>{{ sale.saleTime }}</td>
                <td>{{ sale.actualUnitPrice }}</td>
                <td>{{ sale.soldQuantity }}</td>
                <td>{{ sale.memberID }}</td>
                <td>
                    <button @click="updateSales(sale)">修改</button>
                    <button @click="deleteSales(sale)">删除</button>
                </td>
            </tr>
        </table>
        <div>
            <input v-model="newSale.productID" placeholder="产品ID" />
            <input v-model="newSale.saleTime" placeholder="售出时间" />
            <input v-model="newSale.actualUnitPrice" placeholder="实际单价" />
            <input v-model="newSale.soldQuantity" placeholder="售出数量" />
            <input v-model="newSale.memberID" placeholder="会员ID" />
            <button @click="insertSales(newSale)">添加售出记录</button>
        </div>
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: 'Sales',
    data() {
        return {
            sales: [],
            newSale: {
                productID: '',
                saleTime: '',
                actualUnitPrice: '',
                soldQuantity: '',
                memberID: ''
            }
        }
    },
    created() {
        axios.get('/api/sales/all')
            .then(response => {
                this.sales = response.data;
            });
    },
    methods: {
        updateSales(sale) {
            axios.post('/api/sales/update', sale, {
                headers: {
                    'Content-Type': 'application/json'
                }
            })
                .then(response => {
                    // 更新成功
                });
        },
        insertSales(sale) {
            axios.post('/api/sales/insert', sale, {
                headers: {
                    'Content-Type': 'application/json'
                }
            })
                .then(response => {
                    // 插入成功
                    this.sales.push(sale);
                });
        },
        deleteSales(sale) {
            axios.get(`/api/sales/delete/${sale.salesID}`)
                .then(response => {
                    // 删除成功
                    const index = this.sales.indexOf(sale);
                    this.sales.splice(index, 1);
                });
        }
    }
}
</script>
