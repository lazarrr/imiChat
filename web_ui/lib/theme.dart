import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ------------------------------------------------------------------ brand -- //

/// Primary brand colour — UniChat indigo.
const kIndigo = Color(0xFF4F46E5);

/// Secondary brand colour — the violet end of the signature gradient.
const kViolet = Color(0xFF8B5CF6);

const kSky = Color(0xFF38BDF8);
const kEmerald = Color(0xFF10B981);
const kAmber = Color(0xFFF59E0B);
const kRose = Color(0xFFF43F5E);

/// Signature indigo → violet gradient: logo, primary CTAs, user bubbles.
const kBrandGradient = LinearGradient(
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
  colors: [kIndigo, kViolet],
);

/// Kept under the original name — several widgets reference it directly.
const kUserBubbleGradient = kBrandGradient;

/// Corner-radius scale. Every surface picks from this list so the whole app
/// keeps one consistent curvature instead of ad-hoc numbers.
abstract final class R {
  static const double xs = 8;
  static const double sm = 12;
  static const double md = 16;
  static const double lg = 20;
  static const double xl = 26;
  static const double pill = 999;
}

/// Shared deceleration curve — motion feels settled rather than linear.
const kEase = Cubic(0.22, 1, 0.36, 1);

// ---------------------------------------------------------------- palette -- //

/// Semantic colour tokens for a single brightness.
///
/// Widgets read these through `context.uni` instead of hard-coding hex values,
/// which is what keeps light and dark mode visually consistent.
@immutable
class UniPalette extends ThemeExtension<UniPalette> {
  const UniPalette({
    required this.isDark,
    required this.canvas,
    required this.surface,
    required this.surfaceAlt,
    required this.surfaceSunken,
    required this.border,
    required this.borderStrong,
    required this.textPrimary,
    required this.textSecondary,
    required this.textMuted,
    required this.hover,
    required this.assistantBubble,
    required this.glowA,
    required this.glowB,
    required this.shadow,
  });

  final bool isDark;

  /// Page background.
  final Color canvas;

  /// Cards, header, sidebar, assistant bubbles.
  final Color surface;

  /// Inputs, chips, secondary buttons.
  final Color surfaceAlt;

  /// Code blocks and inset wells.
  final Color surfaceSunken;

  final Color border;
  final Color borderStrong;

  final Color textPrimary;
  final Color textSecondary;
  final Color textMuted;

  /// Wash applied on hover.
  final Color hover;

  final Color assistantBubble;

  /// Two accent colours for the ambient background blobs.
  final Color glowA;
  final Color glowB;

  /// Base colour for elevation shadows.
  final Color shadow;

  static const light = UniPalette(
    isDark: false,
    canvas: Color(0xFFF6F7FC),
    surface: Color(0xFFFFFFFF),
    surfaceAlt: Color(0xFFF1F3FA),
    surfaceSunken: Color(0xFFF5F6FB),
    border: Color(0xFFE6E9F4),
    borderStrong: Color(0xFFD3D8E8),
    textPrimary: Color(0xFF14162A),
    textSecondary: Color(0xFF4B5069),
    textMuted: Color(0xFF7C8299),
    hover: Color(0x0F4F46E5),
    assistantBubble: Color(0xFFFFFFFF),
    glowA: Color(0x33A5B4FC),
    glowB: Color(0x2ED8B4FE),
    shadow: Color(0x1A1B1F3B),
  );

  static const dark = UniPalette(
    isDark: true,
    canvas: Color(0xFF0A0C13),
    surface: Color(0xFF14171F),
    surfaceAlt: Color(0xFF1C202B),
    surfaceSunken: Color(0xFF101319),
    border: Color(0xFF252A38),
    borderStrong: Color(0xFF333A4D),
    textPrimary: Color(0xFFE9EBF3),
    textSecondary: Color(0xFFA9AFC4),
    textMuted: Color(0xFF767D94),
    hover: Color(0x1A8B5CF6),
    assistantBubble: Color(0xFF171B25),
    glowA: Color(0x2E4F46E5),
    glowB: Color(0x248B5CF6),
    shadow: Color(0x66000000),
  );

  @override
  UniPalette copyWith({
    bool? isDark,
    Color? canvas,
    Color? surface,
    Color? surfaceAlt,
    Color? surfaceSunken,
    Color? border,
    Color? borderStrong,
    Color? textPrimary,
    Color? textSecondary,
    Color? textMuted,
    Color? hover,
    Color? assistantBubble,
    Color? glowA,
    Color? glowB,
    Color? shadow,
  }) {
    return UniPalette(
      isDark: isDark ?? this.isDark,
      canvas: canvas ?? this.canvas,
      surface: surface ?? this.surface,
      surfaceAlt: surfaceAlt ?? this.surfaceAlt,
      surfaceSunken: surfaceSunken ?? this.surfaceSunken,
      border: border ?? this.border,
      borderStrong: borderStrong ?? this.borderStrong,
      textPrimary: textPrimary ?? this.textPrimary,
      textSecondary: textSecondary ?? this.textSecondary,
      textMuted: textMuted ?? this.textMuted,
      hover: hover ?? this.hover,
      assistantBubble: assistantBubble ?? this.assistantBubble,
      glowA: glowA ?? this.glowA,
      glowB: glowB ?? this.glowB,
      shadow: shadow ?? this.shadow,
    );
  }

  @override
  UniPalette lerp(ThemeExtension<UniPalette>? other, double t) {
    if (other is! UniPalette) return this;
    Color mix(Color a, Color b) => Color.lerp(a, b, t)!;
    return UniPalette(
      isDark: t < 0.5 ? isDark : other.isDark,
      canvas: mix(canvas, other.canvas),
      surface: mix(surface, other.surface),
      surfaceAlt: mix(surfaceAlt, other.surfaceAlt),
      surfaceSunken: mix(surfaceSunken, other.surfaceSunken),
      border: mix(border, other.border),
      borderStrong: mix(borderStrong, other.borderStrong),
      textPrimary: mix(textPrimary, other.textPrimary),
      textSecondary: mix(textSecondary, other.textSecondary),
      textMuted: mix(textMuted, other.textMuted),
      hover: mix(hover, other.hover),
      assistantBubble: mix(assistantBubble, other.assistantBubble),
      glowA: mix(glowA, other.glowA),
      glowB: mix(glowB, other.glowB),
      shadow: mix(shadow, other.shadow),
    );
  }
}

/// Convenience accessors: `context.uni`, `context.isDark`.
extension UniTheme on BuildContext {
  UniPalette get uni =>
      Theme.of(this).extension<UniPalette>() ?? UniPalette.light;

  bool get isDark => Theme.of(this).brightness == Brightness.dark;
}

// ------------------------------------------------------------------ theme -- //

ThemeData buildTheme(Brightness brightness) {
  final p = brightness == Brightness.dark ? UniPalette.dark : UniPalette.light;

  final scheme = ColorScheme.fromSeed(
    seedColor: kIndigo,
    brightness: brightness,
    dynamicSchemeVariant: DynamicSchemeVariant.vibrant,
  ).copyWith(
    primary: p.isDark ? const Color(0xFF8F90FA) : kIndigo,
    onPrimary: Colors.white,
    surface: p.surface,
    onSurface: p.textPrimary,
    onSurfaceVariant: p.textSecondary,
    surfaceContainerHighest: p.surfaceAlt,
    outline: p.border,
    outlineVariant: p.border,
    shadow: p.shadow,
  );

  final text = GoogleFonts.interTextTheme()
      .apply(bodyColor: p.textPrimary, displayColor: p.textPrimary)
      .copyWith(
        headlineSmall: TextStyle(
          fontSize: 26,
          fontWeight: FontWeight.w800,
          letterSpacing: -0.8,
          height: 1.2,
          color: p.textPrimary,
        ),
        titleMedium: TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
          color: p.textPrimary,
        ),
        titleSmall: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.1,
          color: p.textPrimary,
        ),
        bodyMedium: TextStyle(
          fontSize: 14.5,
          height: 1.5,
          color: p.textPrimary,
        ),
        bodySmall: TextStyle(fontSize: 12.5, color: p.textSecondary),
        labelMedium: TextStyle(
          fontSize: 11.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.6,
          color: p.textSecondary,
        ),
      );

  final base = ThemeData(
    useMaterial3: true,
    brightness: brightness,
    colorScheme: scheme,
    scaffoldBackgroundColor: p.canvas,
    textTheme: text,
    canvasColor: p.canvas,
    dividerColor: p.border,
    splashFactory: InkRipple.splashFactory,
    visualDensity: VisualDensity.standard,
    iconTheme: IconThemeData(color: p.textSecondary, size: 20),
    extensions: [p],
    pageTransitionsTheme: const PageTransitionsTheme(
      builders: {
        TargetPlatform.android: FadeForwardsPageTransitionsBuilder(),
        TargetPlatform.iOS: FadeForwardsPageTransitionsBuilder(),
        TargetPlatform.macOS: FadeForwardsPageTransitionsBuilder(),
        TargetPlatform.windows: FadeForwardsPageTransitionsBuilder(),
        TargetPlatform.linux: FadeForwardsPageTransitionsBuilder(),
      },
    ),
    tooltipTheme: TooltipThemeData(
      waitDuration: const Duration(milliseconds: 400),
      decoration: BoxDecoration(
        color: p.isDark ? const Color(0xFF262B3A) : const Color(0xFF23263A),
        borderRadius: BorderRadius.circular(R.xs),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.22),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      textStyle: const TextStyle(
        color: Colors.white,
        fontSize: 12,
        fontWeight: FontWeight.w500,
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: p.surfaceAlt,
      hintStyle: TextStyle(color: p.textMuted),
      contentPadding:
          const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(R.sm),
        borderSide: BorderSide(color: p.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(R.sm),
        borderSide: BorderSide(color: p.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(R.sm),
        borderSide: const BorderSide(color: kIndigo, width: 1.4),
      ),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: p.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(R.lg),
        side: BorderSide(color: p.border),
      ),
      titleTextStyle: text.titleMedium?.copyWith(fontSize: 17),
      contentTextStyle: text.bodyMedium?.copyWith(color: p.textSecondary),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      backgroundColor: p.isDark ? const Color(0xFF262B3A) : const Color(0xFF23263A),
      contentTextStyle: const TextStyle(
        color: Colors.white,
        fontSize: 13,
        fontWeight: FontWeight.w500,
      ),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(R.sm),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(R.sm),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        textStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13.5),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: p.textSecondary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(R.xs),
        ),
        textStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13.5),
      ),
    ),
    scrollbarTheme: ScrollbarThemeData(
      thickness: const WidgetStatePropertyAll(7),
      radius: const Radius.circular(R.pill),
      thumbColor: WidgetStatePropertyAll(
        p.isDark ? Colors.white.withValues(alpha: 0.16) : Colors.black26,
      ),
      crossAxisMargin: 2,
    ),
  );

  return base.copyWith(
    dividerTheme: DividerThemeData(
      color: p.border,
      thickness: 1,
      space: 1,
    ),
  );
}
