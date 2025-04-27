import { createRouter, createWebHistory } from 'vue-router'
import Statistics from './components/Statistics.vue'
import Sales from './components/Sales.vue'
import Purchase from './components/Purchase.vue'
import Member from './components/Member.vue'
import Staff from './components/Staff.vue'

const routes = [
  { path: '/statistics', component: Statistics },
  { path: '/sales', component: Sales },
  { path: '/purchase', component: Purchase },
  { path: '/member', component: Member },
  { path: '/staff', component: Staff }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
