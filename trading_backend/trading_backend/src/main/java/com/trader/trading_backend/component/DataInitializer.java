package com.trader.trading_backend.component;

import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.HoldingRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.PortfolioRepository;
import com.trader.trading_backend.Repository.User_Management.UserRepository;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Holding;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import com.trader.trading_backend.entity.User_Management.User;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.ArrayList;

@Component
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final UserRepository userRepo;
    private final PortfolioRepository portfolioRepo;
    private final StockRepository stockRepo;
    private final HoldingRepository holdingRepo;
    private final org.springframework.security.crypto.password.PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) throws Exception {
        if (userRepo.count() == 0) {
            System.out.println("🌱 Initializing demo data...");
            
            // 1. Create Demo User
            User user = User.builder()
                    .username("trader_demo")
                    .email("demo@example.com")
                    .passwordHash(passwordEncoder.encode("password"))
                    .build();
            user = userRepo.save(user);

            // 2. Create Default Stocks (if not exist)
            String[] tickers = {"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"};
            for (String t : tickers) {
                if (stockRepo.findByTickerIgnoreCase(t).isEmpty()) {
                    stockRepo.save(Stock.builder()
                            .ticker(t)
                            .companyName(t.split("\\.")[0])
                            .sector("Technology/Finance")
                            .currentPrice(2500.0) // Default price
                            .build());
                }
            }

            // 3. Create Default Portfolio
            Portfolio portfolio = Portfolio.builder()
                    .portfolioName("Default AI Strategy")
                    .initialCapital(new BigDecimal("100000.00"))
                    .currentCashBalance(new BigDecimal("100000.00"))
                    .totalPortfolioValue(new BigDecimal("100000.00"))
                    .user(user)
                    .holdings(new ArrayList<>())
                    .transactions(new ArrayList<>())
                    .build();
            portfolio = portfolioRepo.save(portfolio);
            
            // 4. Create Seed Holdings
            Stock reliance = stockRepo.findByTickerIgnoreCase("RELIANCE.NS").orElseThrow();
            Stock tcs = stockRepo.findByTickerIgnoreCase("TCS.NS").orElseThrow();
            
            holdingRepo.save(Holding.builder()
                    .portfolio(portfolio)
                    .stock(reliance)
                    .quantity(10)
                    .averageBuyPrice(new BigDecimal("2500.00"))
                    .build());
                    
            holdingRepo.save(Holding.builder()
                    .portfolio(portfolio)
                    .stock(tcs)
                    .quantity(5)
                    .averageBuyPrice(new BigDecimal("3200.00"))
                    .build());
            
            System.out.println("✅ Demo data seeded: Portfolio ID 1 created with RELIANCE and TCS holdings.");
        }
    }
}
