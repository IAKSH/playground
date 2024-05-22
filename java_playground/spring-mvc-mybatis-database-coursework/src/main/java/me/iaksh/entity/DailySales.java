package me.iaksh.entity;

import java.math.BigDecimal;
import java.sql.Timestamp;

public class DailySales {
    private Timestamp day;
    private BigDecimal totalSales;

    public BigDecimal getTotalSales() {
        return totalSales;
    }

    public Timestamp getDay() {
        return day;
    }

    public void setDay(Timestamp day) {
        this.day = day;
    }

    public void setTotalSales(BigDecimal totalSales) {
        this.totalSales = totalSales;
    }
}
