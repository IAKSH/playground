package me.iaksh.entity;

import java.math.BigDecimal;

public class TotalExpense {
    private BigDecimal staffExpense;
    private BigDecimal productExpense;

    public BigDecimal getProductExpense() {
        return productExpense;
    }

    public BigDecimal getStaffExpense() {
        return staffExpense;
    }

    public void setProductExpense(BigDecimal productExpense) {
        this.productExpense = productExpense;
    }

    public void setStaffExpense(BigDecimal staffExpense) {
        this.staffExpense = staffExpense;
    }
}
