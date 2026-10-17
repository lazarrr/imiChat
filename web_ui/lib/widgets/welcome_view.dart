import 'package:flutter/material.dart';

import '../theme.dart';
import 'controls.dart';

const _suggestions = [
  (
    icon: Icons.menu_book_rounded,
    text: 'Koliko ESPB bodova ima Uvod u programiranje?',
  ),
  (
    icon: Icons.account_tree_rounded,
    text: 'Koji su smerovi na Prirodno-matematičkom fakultetu?',
  ),
  (
    icon: Icons.rule_rounded,
    text: 'Šta su preduslovi za Analizu 1?',
  ),
  (
    icon: Icons.how_to_reg_rounded,
    text: 'Kako se vrši upis na fakultet?',
  ),
];

/// Landing state shown while a conversation has no messages.
class WelcomeView extends StatelessWidget {
  const WelcomeView({super.key, required this.onSuggestion});

  final ValueChanged<String> onSuggestion;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;

    final gradient = p.isDark
        ? const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFFA5B4FC), Color(0xFFD8B4FE)],
          )
        : kBrandGradient;

    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 36),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const _HeroMark(),
              const SizedBox(height: 26),
              ShaderMask(
                blendMode: BlendMode.srcATop,
                shaderCallback: (rect) => gradient.createShader(rect),
                child: Text(
                  'Dobrodošli u UniChat',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.headlineSmall?.copyWith(
                    fontSize: 32,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -1.1,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 14),
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 480),
                child: Text(
                  'Postavite pitanje o studijama, predmetima i upisu na '
                  'Prirodno-matematičkom fakultetu — odgovori se generišu '
                  'na osnovu zvaničnih dokumenata.',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontSize: 14,
                    height: 1.65,
                    color: p.textSecondary,
                  ),
                ),
              ),
              const SizedBox(height: 22),
              const Wrap(
                alignment: WrapAlignment.center,
                spacing: 8,
                runSpacing: 8,
                children: [
                  Pill(label: 'RAG PRETRAGA', icon: Icons.hub_rounded),
                  Pill(label: 'IZVORI IZ DOKUMENATA', icon: Icons.link_rounded),
                  Pill(label: 'PMF IMI KRAGUJEVAC', icon: Icons.school_rounded),
                ],
              ),
              const SizedBox(height: 34),
              _SuggestionGrid(onSuggestion: onSuggestion),
            ],
          ),
        ),
      ),
    );
  }
}

class _HeroMark extends StatefulWidget {
  const _HeroMark();

  @override
  State<_HeroMark> createState() => _HeroMarkState();
}

class _HeroMarkState extends State<_HeroMark>
    with SingleTickerProviderStateMixin {
  late final AnimationController _float;

  @override
  void initState() {
    super.initState();
    _float = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 5),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _float.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _float,
      builder: (context, child) {
        final t = Curves.easeInOut.transform(_float.value);
        return Transform.translate(offset: Offset(0, -5 * t), child: child);
      },
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Soft halo behind the mark.
          Container(
            width: 132,
            height: 132,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [
                  kIndigo.withValues(alpha: context.isDark ? 0.34 : 0.2),
                  kIndigo.withValues(alpha: 0),
                ],
              ),
            ),
          ),
          Container(
            width: 78,
            height: 78,
            decoration: BoxDecoration(
              gradient: kBrandGradient,
              borderRadius: BorderRadius.circular(24),
              boxShadow: [
                BoxShadow(
                  color: kIndigo.withValues(alpha: 0.45),
                  blurRadius: 34,
                  spreadRadius: -6,
                  offset: const Offset(0, 14),
                ),
              ],
            ),
            child: const Icon(Icons.school_rounded,
                color: Colors.white, size: 38),
          ),
        ],
      ),
    );
  }
}

class _SuggestionGrid extends StatelessWidget {
  const _SuggestionGrid({required this.onSuggestion});

  final ValueChanged<String> onSuggestion;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        const spacing = 12.0;
        final twoColumns = constraints.maxWidth >= 560;
        final itemWidth = twoColumns
            ? (constraints.maxWidth - spacing) / 2
            : constraints.maxWidth;

        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          alignment: WrapAlignment.center,
          children: [
            for (final suggestion in _suggestions)
              SizedBox(
                width: itemWidth,
                child: _SuggestionCard(
                  icon: suggestion.icon,
                  label: suggestion.text,
                  onTap: () => onSuggestion(suggestion.text),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _SuggestionCard extends StatefulWidget {
  const _SuggestionCard({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  State<_SuggestionCard> createState() => _SuggestionCardState();
}

class _SuggestionCardState extends State<_SuggestionCard> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        curve: kEase,
        transform: Matrix4.translationValues(0, _hovering ? -3 : 0, 0),
        decoration: BoxDecoration(
          color: p.surface,
          borderRadius: BorderRadius.circular(R.md),
          border: Border.all(
            color: _hovering ? kIndigo.withValues(alpha: 0.45) : p.border,
          ),
          boxShadow: [
            BoxShadow(
              color: _hovering
                  ? kIndigo.withValues(alpha: p.isDark ? 0.26 : 0.16)
                  : p.shadow,
              blurRadius: _hovering ? 22 : 14,
              spreadRadius: -8,
              offset: Offset(0, _hovering ? 10 : 6),
            ),
          ],
        ),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            onTap: widget.onTap,
            borderRadius: BorderRadius.circular(R.md),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 13, 12, 13),
              child: Row(
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    curve: kEase,
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      color: kIndigo.withValues(
                        alpha: _hovering
                            ? (p.isDark ? 0.3 : 0.14)
                            : (p.isDark ? 0.18 : 0.08),
                      ),
                      borderRadius: BorderRadius.circular(R.xs + 1),
                    ),
                    child: Icon(
                      widget.icon,
                      size: 16,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      widget.label,
                      style: TextStyle(
                        fontSize: 13,
                        height: 1.4,
                        fontWeight: FontWeight.w600,
                        color: p.textPrimary,
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  AnimatedSlide(
                    duration: const Duration(milliseconds: 180),
                    curve: kEase,
                    offset: Offset(_hovering ? 0.12 : 0, 0),
                    child: Icon(
                      Icons.arrow_forward_rounded,
                      size: 16,
                      color: _hovering
                          ? Theme.of(context).colorScheme.primary
                          : p.textMuted,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
