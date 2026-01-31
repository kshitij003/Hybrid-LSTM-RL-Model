package com.trader.trading_backend.Repository.User_Management;

import com.trader.trading_backend.entity.User_Management.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User,Long> {
}
