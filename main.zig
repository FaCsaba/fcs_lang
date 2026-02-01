const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Tokenizer = struct {
    src: []const u8 = &.{},
    byte_idx: usize = 0,

    const Token = struct {
        repr: []const u8,
        kind: Kind,

        const Keywords = std.StaticStringMap(Kind).initComptime(.{
            .{ "legyen", .keyword_legyen },
        });

        const Kind = enum {
            invalid,
            symbol,
            number,
            float,
            string,
            equals,
            lparen,
            rparen,
            newline,
            plus,
            keyword_legyen,
        };
    };

    fn nextCodepoint(t: *Tokenizer) u21 {
        var iter = std.unicode.Utf8Iterator{ .bytes = t.src, .i = t.byte_idx };
        defer t.byte_idx = iter.i;
        return iter.nextCodepoint() orelse '\x00';
    }

    fn peekCodepoint(t: Tokenizer) u21 {
        var iter = std.unicode.Utf8Iterator{ .bytes = t.src, .i = t.byte_idx };
        return iter.nextCodepoint() orelse '\x00';
    }

    const State = enum {
        start,
        eat_whitespace,
        number,
        number_dot,
        number_float,
        symbol,
    };

    fn repr(t: *const Tokenizer, start: usize) []const u8 {
        return t.src[start..t.byte_idx];
    }

    fn nextToken(t: *Tokenizer) ?Token {
        var start = t.byte_idx;

        loop: switch (State.start) {
            .start => switch (t.peekCodepoint()) {
                0 => return null,
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number;
                },
                '\n' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .newline };
                },
                ' ', '\t', 11...'\r' => continue :loop .eat_whitespace,
                '(' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .lparen };
                },
                ')' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .rparen };
                },
                '=' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .equals };
                },
                '+' => {
                    _ = t.nextCodepoint();
                    return Token{ .repr = t.repr(start), .kind = .plus };
                },
                'A'...'Z', 'a'...'z', 128...std.math.maxInt(u21) => {
                    _ = t.nextCodepoint();
                    continue :loop .symbol;
                },
                else => return Token{ .repr = t.repr(start), .kind = .invalid },
            },
            .eat_whitespace => switch (t.peekCodepoint()) {
                '\n' => continue :loop .start,
                ' ', '\t', 11...'\r' => {
                    start += 1;
                    _ = t.nextCodepoint();
                    continue :loop .eat_whitespace;
                },
                else => continue :loop .start,
            },
            .number => switch (t.peekCodepoint()) {
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number;
                },
                '.' => {
                    _ = t.nextCodepoint();
                    continue :loop .number_dot;
                },
                else => return Token{ .repr = t.repr(start), .kind = .number },
            },
            .number_dot => switch (t.peekCodepoint()) {
                '0'...'9' => continue :loop .number_float,
                else => return Token{ .repr = t.repr(start), .kind = .number },
            },
            .number_float => switch (t.peekCodepoint()) {
                '0'...'9' => {
                    _ = t.nextCodepoint();
                    continue :loop .number_float;
                },
                else => return Token{ .repr = t.repr(start), .kind = .float },
            },
            .symbol => switch (t.peekCodepoint()) {
                'A'...'Z', 'a'...'z', 128...std.math.maxInt(u21) => {
                    _ = t.nextCodepoint();
                    continue :loop .symbol;
                },
                else => {
                    const symbol = t.repr(start);
                    if (Token.Keywords.has(symbol)) {
                        return Token{ .repr = symbol, .kind = Token.Keywords.get(symbol).? };
                    } else {
                        return Token{ .repr = symbol, .kind = .symbol };
                    }
                },
            },
        }
    }

    fn peekToken(t: *Tokenizer) ?Token {
        const byte_idx = t.byte_idx;
        defer t.byte_idx = byte_idx;
        return t.nextToken();
    }
};

fn expectTokenEqual(repr: []const u8, kind: Tokenizer.Token.Kind, token: ?Tokenizer.Token) !void {
    try std.testing.expectEqualStrings(repr, token.?.repr);
    try std.testing.expectEqual(kind, token.?.kind);
}
test "Tokenizer" {
    var tz1 = Tokenizer{ .src = "legyen kutya = 123" };
    try expectTokenEqual("legyen", .keyword_legyen, tz1.nextToken());
    try expectTokenEqual("kutya", .symbol, tz1.nextToken());
    try expectTokenEqual("=", .equals, tz1.nextToken());
    try expectTokenEqual("123", .number, tz1.nextToken());

    var tz2 = Tokenizer{ .src = "alma\n123.456\nbűnöző" };
    try expectTokenEqual("alma", .symbol, tz2.nextToken());
    try expectTokenEqual("\n", .newline, tz2.nextToken());
    try expectTokenEqual("123.456", .float, tz2.nextToken());
    try expectTokenEqual("\n", .newline, tz2.nextToken());
    try expectTokenEqual("bűnöző", .symbol, tz2.nextToken());
}

// legyen kutya: szöveg = "vakk"
// legyen macska: szöveg = "miáú"
// legyen alma: szöveg | semmi = ""
// kutya = "vakk vakk"
// írd(kutya)
