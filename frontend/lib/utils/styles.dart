// lib/utils/styles.dart
import 'package:flutter/material.dart';

// Color Palette
class AppColors {
  static const primaryBlue = Color(0xFF1A73E8);
  static const darkBlue = Color(0xFF0D47A1);
  static const electricCyan = Color(0xFF00FFFF);
  static const neonGreen = Color(0xFF39FF14);
  static const midnightBlack = Color(0xFF121212);
  static const coolGray = Color(0xFFB0BEC5);
  static const white = Color(0xFFFFFFFF);
  static const purpleAccent = Color(0xFF7C4DFF);
}

// Text Styles
class AppTextStyles {
  static const bodyMedium = TextStyle(
    color: AppColors.white,
    fontSize: 16,
  );

  static const titleLarge = TextStyle(
    color: AppColors.white,
    fontSize: 24,
    fontWeight: FontWeight.bold,
  );

  static const buttonText = TextStyle(
    color: AppColors.white,
    fontSize: 16,
    fontWeight: FontWeight.bold,
  );

  static const dialogTitle = TextStyle(
    color: AppColors.white,
    fontSize: 20,
    fontWeight: FontWeight.bold,
  );

  static const dialogContent = TextStyle(
    color: AppColors.coolGray,
    fontSize: 16,
  );
}

// App Theme
final appTheme = ThemeData(
  primaryColor: AppColors.primaryBlue,
  colorScheme: ColorScheme.light(
    primary: AppColors.primaryBlue,
    secondary: AppColors.purpleAccent,
    surface: AppColors.white, // Replaced background
    onPrimary: AppColors.white,
    onSecondary: AppColors.white,
    onSurface: AppColors.midnightBlack, // Replaced onBackground
    tertiary: AppColors.electricCyan,
    tertiaryContainer: AppColors.neonGreen,
  ),
  textTheme: TextTheme(
    bodyMedium: AppTextStyles.bodyMedium,
    titleLarge: AppTextStyles.titleLarge,
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: AppColors.primaryBlue,
      foregroundColor: AppColors.white,
      padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
      ),
    ),
  ),
  appBarTheme: AppBarTheme(
    backgroundColor: AppColors.darkBlue,
    titleTextStyle: AppTextStyles.titleLarge,
  ),
  cardTheme: CardTheme(
    color: AppColors.darkBlue,
    elevation: 4,
    shadowColor: Colors.black26,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(8),
    ),
  ),
  inputDecorationTheme: InputDecorationTheme(
    filled: true,
    fillColor: AppColors.coolGray,
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(8),
      borderSide: BorderSide.none,
    ),
  ),
);


