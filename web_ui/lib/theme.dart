import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

const kIndigo = Color(0xFF4F46E5);
const kViolet = Color(0xFF8B5CF6);

const kUserBubbleGradient = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [kIndigo, kViolet],
);

ThemeData buildTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final scheme = ColorScheme.fromSeed(
    seedColor: kIndigo,
    brightness: brightness,
  );

  final base = ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor:
        isDark ? const Color(0xFF0E1016) : const Color(0xFFF6F7FB),
    textTheme: GoogleFonts.interTextTheme().apply(
      bodyColor: isDark ? const Color(0xFFE6E8EF) : const Color(0xFF1B1D28),
      displayColor: isDark ? const Color(0xFFE6E8EF) : const Color(0xFF1B1D28),
    ),
    tooltipTheme: TooltipThemeData(
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF262A38) : const Color(0xFF2B2D3A),
        borderRadius: BorderRadius.circular(8),
      ),
      textStyle: const TextStyle(color: Colors.white, fontSize: 12),
    ),
  );

  return base.copyWith(
    dividerTheme: DividerThemeData(
      color: isDark ? const Color(0xFF232637) : const Color(0xFFE8EAF2),
      thickness: 1,
      space: 1,
    ),
    scrollbarTheme: ScrollbarThemeData(
      thumbColor: WidgetStatePropertyAll(
        isDark ? Colors.white.withValues(alpha: 0.18) : Colors.black26,
      ),
    ),
  );
}
