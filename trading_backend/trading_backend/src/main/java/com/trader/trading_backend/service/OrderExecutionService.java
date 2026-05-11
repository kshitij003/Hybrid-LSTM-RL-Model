package com.trader.trading_backend.service;

import com.trader.trading_backend.Enum.TransactionType;
import com.trader.trading_backend.Repository.Market_Data_Management.StockPriceRepository;
import com.trader.trading_backend.Repository.Market_Data_Management.StockRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.HoldingRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.PortfolioRepository;
import com.trader.trading_backend.Repository.Portfolio_Trading_Engine.TransactionRepository;
import com.trader.trading_backend.dto.OrderRequestDTO;
import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Holding;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Transaction;
import com.trader.trading_backend.controller.DashboardController;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class OrderExecutionService {

    private final PortfolioRepository portfolioRepo;
    private final StockRepository stockRepo;
    private final StockPriceRepository priceRepo;
    private final TransactionRepository transactionRepo;
    private final HoldingRepository holdingRepo;

    // Simulation Config: 0.1% fee per trade
    private static final BigDecimal TRANSACTION_FEE_PERCENT = BigDecimal.valueOf(0.001);

    /**
     * MAIN METHOD: Execute a single order safely.
     * Returns true if successful, false if rejected (e.g. insufficient funds).
     */
    @Transactional
    public boolean executeOrder(OrderRequestDTO request) {
        System.out.println("⚡ Executing Order: {} {} shares of {}"+request.getType()+request.getQuantity()+request.getTicker());

        // 1. Fetch Fresh Entities (Ensure latest state)
        Portfolio portfolio = portfolioRepo.findById(request.getPortfolioId())
                .orElseThrow(() -> new RuntimeException("Portfolio not found"));

        Stock stock = stockRepo.findByTickerIgnoreCase(request.getTicker())
                .orElseThrow(() -> new RuntimeException("Stock not found"));

        // 2. Get Real-Time Price
        BigDecimal currentPrice = getLatestPrice(stock);

        // 3. Handle LIVE Trading Bridge (External API)
        if (DashboardController.isLiveTradingMode) {
            triggerLiveTrade(request.getTicker(), request.getType().name(), request.getQuantity());
        }

        // 4. Calculate Financials
        BigDecimal tradeAmount = currentPrice.multiply(BigDecimal.valueOf(request.getQuantity()));
        BigDecimal fee = tradeAmount.multiply(TRANSACTION_FEE_PERCENT);
        BigDecimal finalAmount = tradeAmount.add(fee); // Total cost to user

        // 5. Route to Buy or Sell Logic
        if (request.getType() == TransactionType.BUY) {
            return processBuy(portfolio, stock, request.getQuantity(), currentPrice, tradeAmount, fee);
        } else {
            return processSell(portfolio, stock, request.getQuantity(), currentPrice, tradeAmount, fee);
        }
    }

    // --- BUY LOGIC ---
    private boolean processBuy(Portfolio portfolio, Stock stock, int quantity, BigDecimal price, BigDecimal rawAmount, BigDecimal fee) {
        BigDecimal totalCost = rawAmount.add(fee);

        // A. Validation: Do we have enough cash?
        if (portfolio.getCurrentCashBalance().compareTo(totalCost) < 0) {
            System.out.println("❌ Order Rejected: Insufficient Funds. Need {}, Have {}"+totalCost+portfolio.getCurrentCashBalance());
            return false;
        }

        // B. Update Cash
        portfolio.setCurrentCashBalance(portfolio.getCurrentCashBalance().subtract(totalCost));

        // C. Update Holdings (Add Shares)
        Holding holding = holdingRepo.findByPortfolioAndStock(portfolio, stock)
                .orElse(Holding.builder()
                        .portfolio(portfolio)
                        .stock(stock)
                        .quantity(0)
                        .averageBuyPrice(BigDecimal.ZERO)
                        .build());

        // Recalculate Average Buy Price (Weighted Average)
        // NewAvg = ((OldQty * OldAvg) + (NewQty * NewPrice)) / TotalQty
        BigDecimal oldTotalVal = holding.getAverageBuyPrice().multiply(BigDecimal.valueOf(holding.getQuantity()));
        BigDecimal newTotalVal = oldTotalVal.add(rawAmount); // Use raw amount for avg price, exclude fee
        int newQty = holding.getQuantity() + quantity;

        holding.setQuantity(newQty);
        holding.setAverageBuyPrice(newTotalVal.divide(BigDecimal.valueOf(newQty), 4, BigDecimal.ROUND_HALF_UP));

        // D. Save Changes
        holdingRepo.save(holding);
        portfolioRepo.save(portfolio);

        // E. Log Transaction
        recordTransaction(portfolio, stock, TransactionType.BUY, quantity, price, fee, totalCost);

        return true;
    }

    // --- SELL LOGIC ---
    private boolean processSell(Portfolio portfolio, Stock stock, int quantity, BigDecimal price, BigDecimal rawAmount, BigDecimal fee) {
        // A. Validation: Do we have enough shares?
        Holding holding = holdingRepo.findByPortfolioAndStock(portfolio, stock)
                .orElse(null);

        if (holding == null || holding.getQuantity() < quantity) {
            System.out.println("❌ Order Rejected: Insufficient Shares. Need {}, Have {}"+quantity+(holding == null ? 0 : holding.getQuantity()));
            return false;
        }

        // B. Calculate Net Cash Received (Amount - Fee)
        BigDecimal netCashReceived = rawAmount.subtract(fee);

        // C. Update Cash
        portfolio.setCurrentCashBalance(portfolio.getCurrentCashBalance().add(netCashReceived));

        // D. Update Holdings (Remove Shares)
        int remainingQty = holding.getQuantity() - quantity;
        if (remainingQty == 0) {
            holdingRepo.delete(holding); // Remove row if 0 shares left
        } else {
            holding.setQuantity(remainingQty);
            holdingRepo.save(holding);
        }
        portfolioRepo.save(portfolio);

        // E. Log Transaction
        recordTransaction(portfolio, stock, TransactionType.SELL, quantity, price, fee, netCashReceived);

        return true;
    }

    // --- HELPER: Audit Trail ---
    private void recordTransaction(Portfolio p, Stock s, TransactionType type, int qty, BigDecimal price, BigDecimal fee, BigDecimal total) {
        Transaction tx = Transaction.builder()
                .portfolio(p)
                .stock(s)
                .type(type)
                .transactionDate(LocalDateTime.now())
                .quantity(qty)
                .priceAtTransaction(price)
                // .transactionFee(fee) // Add this field to your Transaction Entity if you haven't yet
                .totalAmount(total)
                .build();

        transactionRepo.save(tx);
        System.out.println("✅ Transaction Recorded: {} {} | Total: ${}"+type+s.getTicker()+total);
    }

    private BigDecimal getLatestPrice(Stock stock) {
        // Fetch most recent price (Limit 1)
        var latest = priceRepo.findByStockIdOrderByTimestampDesc(stock.getId(), org.springframework.data.domain.PageRequest.of(0, 1));
        
        if (latest.isEmpty()) {
            System.out.println("⚠️ Price history missing for " + stock.getTicker() + ". Using currentPrice field as fallback.");
            if (stock.getCurrentPrice() != null) {
                return BigDecimal.valueOf(stock.getCurrentPrice());
            }
            return BigDecimal.valueOf(100.0); // Final fallback for demo
        }
        
        return latest.get(0).getClosePrice();
    }

    private void triggerLiveTrade(String ticker, String action, int quantity) {
        try {
            System.out.println("🚀 [LIVE BRIDGE] Sending order to Groww Bridge: {} {} shares of {}"+action+quantity+ticker);
            RestTemplate restTemplate = new RestTemplate();
            String url = "http://localhost:8000/api/trade";
            
            Map<String, Object> request = new HashMap<>();
            request.put("ticker", ticker);
            request.put("type", action);
            request.put("quantity", quantity);
            
            restTemplate.postForEntity(url, request, String.class);
            System.out.println("✅ [LIVE BRIDGE] Groww acknowledge received");
        } catch (Exception e) {
            System.err.println("❌ [LIVE BRIDGE] FAILED to call Groww API: " + e.getMessage());
            // In a production app, we would throw an exception here to rollback the transaction
        }
    }
}

