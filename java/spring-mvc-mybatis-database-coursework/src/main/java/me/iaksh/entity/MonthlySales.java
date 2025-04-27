package me.iaksh.entity;

import java.math.BigDecimal;

public class MonthlySales {
    private String month;
    private BigDecimal totalSales;

    public BigDecimal getTotalSales() {
        return totalSales;
    }

    public String getMonth() {
        return month;
    }

    public void setTotalSales(BigDecimal totalSales) {
        this.totalSales = totalSales;
    }

    public void setMonth(String month) {
        this.month = month;
    }
}
