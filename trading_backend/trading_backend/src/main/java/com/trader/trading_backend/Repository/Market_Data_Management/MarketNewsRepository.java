package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.dto.DailySentimentDTO;
import com.trader.trading_backend.entity.Market_Data_Management.MarketNews;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface  MarketNewsRepository extends JpaRepository<MarketNews,Long> {

    @Query("SELECT new com.trader.trading_backend.dto.DailySentimentDTO(n.stock.id, CAST(n.publishedAt AS LocalDate), AVG(n.sentimentScore), COUNT(n)) " +
            "FROM MarketNews n " +
            "WHERE n.publishedAt < :cutoffDate " +
            "GROUP BY n.stock.id, CAST(n.publishedAt AS LocalDate)")
    List<DailySentimentDTO> findSentimentAggregates(@Param("cutoffDate") LocalDateTime cutoffDate);

    @Modifying
    @Query("DELETE FROM MarketNews n WHERE n.publishedAt < :cutoffDate")
    void deleteAllBefore(@Param("cutoffDate") LocalDateTime cutoffDate);

    Double findAverageSentimentByStockIdAndPublishedAt(Long stockId, LocalDate date);
}
