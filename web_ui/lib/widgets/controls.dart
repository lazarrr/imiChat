import 'package:flutter/material.dart';

import '../theme.dart';

/// Animates [child] on pointer hover — the small lift that makes buttons and
/// cards feel alive on the web.
class HoverScale extends StatefulWidget {
  const HoverScale({
    super.key,
    required this.child,
    this.scale = 1.02,
    this.lift = -2,
    this.enabled = true,
  });

  final Widget child;
  final double scale;
  final double lift;
  final bool enabled;

  @override
  State<HoverScale> createState() => _HoverScaleState();
}

class _HoverScaleState extends State<HoverScale> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final active = widget.enabled && _hovering;
    return MouseRegion(
      cursor: widget.enabled
          ? SystemMouseCursors.click
          : SystemMouseCursors.basic,
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        curve: kEase,
        transform: Matrix4.translationValues(
          0,
          active ? widget.lift : 0,
          0,
        )..scaleByDouble(
            active ? widget.scale : 1,
            active ? widget.scale : 1,
            1,
            1,
          ),
        transformAlignment: Alignment.center,
        child: widget.child,
      ),
    );
  }
}

/// Compact icon button with a hover wash and an optional selected state.
class IconAction extends StatefulWidget {
  const IconAction({
    super.key,
    required this.icon,
    this.onPressed,
    this.tooltip,
    this.iconSize = 18,
    this.tint,
    this.selected = false,
    this.padding = const EdgeInsets.all(8),
  });

  final IconData icon;
  final VoidCallback? onPressed;
  final String? tooltip;
  final double iconSize;
  final Color? tint;
  final bool selected;
  final EdgeInsets padding;

  @override
  State<IconAction> createState() => _IconActionState();
}

class _IconActionState extends State<IconAction> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;
    final enabled = widget.onPressed != null;

    final color = widget.tint ??
        (widget.selected ? theme.colorScheme.primary : p.textSecondary);

    final Color background;
    if (!enabled) {
      background = Colors.transparent;
    } else if (widget.selected) {
      background = kIndigo.withValues(alpha: _hovering ? 0.22 : 0.14);
    } else {
      background = _hovering ? p.hover : Colors.transparent;
    }

    final button = MouseRegion(
      cursor: enabled ? SystemMouseCursors.click : SystemMouseCursors.basic,
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 140),
        curve: kEase,
        decoration: BoxDecoration(
          color: background,
          borderRadius: BorderRadius.circular(R.xs + 2),
        ),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            onTap: widget.onPressed,
            borderRadius: BorderRadius.circular(R.xs + 2),
            child: Padding(
              padding: widget.padding,
              child: Icon(
                widget.icon,
                size: widget.iconSize,
                color: enabled ? color : color.withValues(alpha: 0.35),
              ),
            ),
          ),
        ),
      ),
    );

    if (widget.tooltip == null) return button;
    return Tooltip(message: widget.tooltip!, child: button);
  }
}

/// Primary call-to-action: indigo → violet gradient with a coloured glow that
/// intensifies on hover.
class GradientButton extends StatefulWidget {
  const GradientButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.icon,
    this.expand = true,
    this.height = 44,
  });

  final String label;
  final VoidCallback? onPressed;
  final IconData? icon;
  final bool expand;
  final double height;

  @override
  State<GradientButton> createState() => _GradientButtonState();
}

class _GradientButtonState extends State<GradientButton> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final enabled = widget.onPressed != null;
    final hovered = enabled && _hovering;

    final child = MouseRegion(
      cursor: enabled ? SystemMouseCursors.click : SystemMouseCursors.basic,
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        curve: kEase,
        height: widget.height,
        decoration: BoxDecoration(
          gradient: kBrandGradient,
          borderRadius: BorderRadius.circular(R.sm + 2),
          boxShadow: [
            BoxShadow(
              color: kIndigo.withValues(alpha: hovered ? 0.5 : 0.3),
              blurRadius: hovered ? 22 : 14,
              spreadRadius: hovered ? -1 : -3,
              offset: Offset(0, hovered ? 6 : 4),
            ),
          ],
        ),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            onTap: widget.onPressed,
            borderRadius: BorderRadius.circular(R.sm + 2),
            child: Center(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (widget.icon != null) ...[
                    Icon(widget.icon, color: Colors.white, size: 18),
                    const SizedBox(width: 8),
                  ],
                  Flexible(
                    child: Text(
                      widget.label,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                        fontSize: 13.5,
                        letterSpacing: -0.1,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );

    return widget.expand ? SizedBox(width: double.infinity, child: child) : child;
  }
}

/// Small uppercase capsule used for metadata ("RAG", "Izvori", model names).
class Pill extends StatelessWidget {
  const Pill({
    super.key,
    required this.label,
    this.icon,
    this.color,
    this.dense = false,
  });

  final String label;
  final IconData? icon;
  final Color? color;
  final bool dense;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    final accent = color ?? Theme.of(context).colorScheme.primary;

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: dense ? 8 : 10,
        vertical: dense ? 3.5 : 5,
      ),
      decoration: BoxDecoration(
        color: accent.withValues(alpha: p.isDark ? 0.18 : 0.09),
        borderRadius: BorderRadius.circular(R.pill),
        border: Border.all(color: accent.withValues(alpha: 0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: dense ? 11 : 13, color: accent),
            const SizedBox(width: 5),
          ],
          Text(
            label,
            style: TextStyle(
              fontSize: dense ? 10.5 : 11.5,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.1,
              color: p.isDark ? accent.withValues(alpha: 0.95) : accent,
            ),
          ),
        ],
      ),
    );
  }
}
