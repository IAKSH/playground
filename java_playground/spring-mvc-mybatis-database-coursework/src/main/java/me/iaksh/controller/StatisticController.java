package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.service.*;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/statistics")
public class StatisticController {
    private static DailySalesService dailySalesService;
    private static MonthlySalesService monthlySalesService;
    private static YearlySalesService yearlySalesService;
    private static TotalIncomeService totalIncomeService;
    private static TotalExpenseService totalExpenseService;

    public static void setDailySalesService(DailySalesService dailySalesService) {
        StatisticController.dailySalesService = dailySalesService;
    }

    public static void setMonthlySalesService(MonthlySalesService monthlySalesService) {
        StatisticController.monthlySalesService = monthlySalesService;
    }

    public static void setTotalExpenseService(TotalExpenseService totalExpenseService) {
        StatisticController.totalExpenseService = totalExpenseService;
    }

    public static void setTotalIncomeService(TotalIncomeService totalIncomeService) {
        StatisticController.totalIncomeService = totalIncomeService;
    }

    public static void setYearlySalesService(YearlySalesService yearlySalesService) {
        StatisticController.yearlySalesService = yearlySalesService;
    }

    @GetMapping(value = "/sales/daily",
            produces = "application/json;charset=UTF-8")
    public String getDailySales() {
        return JSON.toJSONString(dailySalesService.get());
    }

    @GetMapping(value = "/sales/monthly",
            produces = "application/json;charset=UTF-8")
    public String getMonthlySales() {
        return JSON.toJSONString(monthlySalesService.get());
    }

    @GetMapping(value = "/sales/yearly",
            produces = "application/json;charset=UTF-8")
    public String getYearlySales() {
        return JSON.toJSONString(yearlySalesService.get());
    }

    @GetMapping(value = "/expense",
            produces = "application/json;charset=UTF-8")
    public String getTotalExpense() {
        return JSON.toJSONString(totalExpenseService.get());
    }

    @GetMapping(value = "/income",
            produces = "application/json;charset=UTF-8")
    public String getTotalIncome() {
        return JSON.toJSONString(totalIncomeService.get());
    }
}
